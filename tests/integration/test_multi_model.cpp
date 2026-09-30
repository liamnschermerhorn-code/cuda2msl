#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>

// ===========================================================================
// Multi-Model Architecture Tests
//
//   1. Mistral 7B   -- Sliding window attention (INT8)
//   2. Mixtral 8x7B  -- Mixture of Experts, top-2 routing (FP32)
//   3. DeepSeek-V2   -- Multi-head Latent Attention + MoE (INT8)
//   4. Whisper Large  -- Encoder-decoder with cross-attention (FP32)
//   5. BERT Large     -- Bidirectional encoder, no causal mask (FP32)
// ===========================================================================

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "FAIL: %s at %s:%d\n", #call, __FILE__, __LINE__); return false; } } while(0)
#define CUBLAS_CHECK(call) do { cublasStatus_t st = (call); if (st != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "FAIL: %s at %s:%d\n", #call, __FILE__, __LINE__); return false; } } while(0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
};

static void fill_random(float* p, size_t n, float scale = 0.02f) {
    for (size_t i = 0; i < n; i++)
        p[i] = scale * ((float)(rand() % 2000 - 1000) / 1000.0f);
}

static bool all_finite(const float* p, size_t n) {
    for (size_t i = 0; i < n; i++) if (!std::isfinite(p[i])) return false;
    return true;
}

static bool alloc_weight(float** p, size_t count, float scale = 0.02f) {
    if (cudaMalloc((void**)p, count * sizeof(float)) != cudaSuccess) return false;
    fill_random(*p, count, scale);
    return true;
}

static void rmsnorm(float* out, const float* x, const float* w, int d, int tok) {
    for (int t = 0; t < tok; t++) {
        const float* r = x + t * d; float* o = out + t * d;
        float ss = 0;
        for (int i = 0; i < d; i++) ss += r[i] * r[i];
        ss = 1.0f / sqrtf(ss / d + 1e-5f);
        for (int i = 0; i < d; i++) o[i] = r[i] * ss * w[i];
    }
}

static void layernorm(float* out, const float* x, const float* g, const float* b, int d, int tok) {
    for (int t = 0; t < tok; t++) {
        const float* r = x + t * d; float* o = out + t * d;
        float mean = 0; for (int i = 0; i < d; i++) mean += r[i]; mean /= d;
        float var = 0; for (int i = 0; i < d; i++) var += (r[i]-mean)*(r[i]-mean); var /= d;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int i = 0; i < d; i++) o[i] = (r[i] - mean) * inv * g[i] + b[i];
    }
}

static void swiglu(float* out, const float* gate, const float* up, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float g = gate[i]; out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

static void gelu_act(float* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float x = data[i];
        data[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

static void rope(float* qk, int tok, int hd, int start) {
    for (int t = 0; t < tok; t++) {
        int pos = start + t;
        for (int i = 0; i < hd; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / (float)hd);
            float a = pos * freq, c = cosf(a), s = sinf(a);
            float v0 = qk[t * hd + i], v1 = qk[t * hd + i + 1];
            qk[t * hd + i]     = v0 * c - v1 * s;
            qk[t * hd + i + 1] = v0 * s + v1 * c;
        }
    }
}

static void softmax_row(float* x, int len) {
    float mx = *std::max_element(x, x + len); float sum = 0;
    for (int i = 0; i < len; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < len; i++) x[i] /= sum;
}

static void softmax_causal(float* sc, int qpos, int kvlen) {
    for (int j = qpos + 1; j < kvlen; j++) sc[j] = -1e10f;
    softmax_row(sc, kvlen);
}

static void softmax_window(float* sc, int qpos, int kvlen, int win) {
    int lo = std::max(0, qpos - win + 1);
    for (int j = 0; j < kvlen; j++)
        if (j > qpos || j < lo) sc[j] = -1e10f;
    softmax_row(sc, kvlen);
}

// ---------------------------------------------------------------------------
// INT8 quantized weight helpers
// ---------------------------------------------------------------------------
struct QWeight { void* data; void* scales; int rows, cols; };

static size_t qw_bytes(const QWeight& w) {
    return (size_t)w.rows * w.cols + (size_t)w.rows * sizeof(float);
}

static bool alloc_qw(QWeight& w, int rows, int cols) {
    w.rows = rows; w.cols = cols;
    size_t dsz = (size_t)rows * cols;
    if (cudaMalloc(&w.data, dsz) != cudaSuccess) return false;
    if (cudaMalloc(&w.scales, rows * sizeof(float)) != cudaSuccess) { cudaFree(w.data); return false; }
    cudaMemset(w.data, 0, dsz);
    int stripe = std::min(cols, 512);
    std::vector<uint8_t> buf(stripe);
    for (int i = 0; i < stripe; i++) buf[i] = (uint8_t)(rand() % 256);
    cudaMemcpy(w.data, buf.data(), stripe, cudaMemcpyHostToDevice);
    std::vector<float> sc(rows, 0.001f);
    for (int i = 0; i < std::min(rows, 64); i++) sc[i] = 0.01f * ((rand() % 100 + 50) / 100.0f);
    cudaMemcpy(w.scales, sc.data(), rows * sizeof(float), cudaMemcpyHostToDevice);
    return true;
}

static void free_qw(QWeight& w) { cudaFree(w.data); cudaFree(w.scales); }

static void dequant_block(const QWeight& w, float* out, int col_start, int ncols) {
    const int8_t* d = (const int8_t*)w.data;
    const float* s = (const float*)w.scales;
    for (int r = 0; r < w.rows; r++) {
        float scale = s[r];
        const int8_t* row = d + (size_t)r * w.cols + col_start;
        float* orow = out + (size_t)r * ncols;
        for (int c = 0; c < ncols; c++) orow[c] = (float)row[c] * scale;
    }
}

static cublasStatus_t qgemm(cublasHandle_t h, int M, const float* X, const QWeight& W,
                             float* Y, float* deq, int chunk) {
    int K = W.rows, N = W.cols; float one = 1.0f, zero = 0.0f;
    for (int c = 0; c < N; c += chunk) {
        int nc = std::min(chunk, N - c);
        dequant_block(W, deq, c, nc);
        cublasStatus_t st = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
            M, nc, K, &one, X, M, deq, K, &zero, Y + (size_t)c * M, M);
        if (st != CUBLAS_STATUS_SUCCESS) return st;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// =====================================================================
// 1. Mistral 7B — Sliding Window Attention (INT8)
// =====================================================================
static bool test_mistral_7b() {
    const int d = 4096, d_ff = 14336, nl = 32, seq = 128, win = 64;
    size_t ppl = 4*(size_t)d*d + 2*(size_t)d*d_ff + (size_t)d_ff*d;
    printf("  %.1fB params, d=%d, d_ff=%d, %d layers, seq=%d, window=%d\n",
           ppl * nl / 1e9, d, d_ff, nl, seq, win);

    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas)); srand(7777);

    QWeight Wq, Wk, Wv, Wo, Wg, Wu, Wd;
    if (!alloc_qw(Wq,d,d) || !alloc_qw(Wk,d,d) || !alloc_qw(Wv,d,d) || !alloc_qw(Wo,d,d) ||
        !alloc_qw(Wg,d,d_ff) || !alloc_qw(Wu,d,d_ff) || !alloc_qw(Wd,d_ff,d))
        { fprintf(stderr,"  OOM\n"); return false; }

    float* rms_w; CUDA_CHECK(cudaMalloc((void**)&rms_w, d*4));
    { std::vector<float> o(d,1.0f); cudaMemcpy(rms_w,o.data(),d*4,cudaMemcpyHostToDevice); }

    size_t de = (size_t)d*d; float* deq;
    CUDA_CHECK(cudaMalloc((void**)&deq, de*4));
    int ca = d, cf = (int)(de/d), cd = (int)(de/d_ff);

    size_t xs = (size_t)seq*d, fs = (size_t)seq*d_ff;
    float *x,*nm,*q,*k,*v,*sc,*ao,*ga,*ua;
    CUDA_CHECK(cudaMalloc((void**)&x,xs*4));  CUDA_CHECK(cudaMalloc((void**)&nm,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&q,xs*4));  CUDA_CHECK(cudaMalloc((void**)&k,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&v,xs*4));  CUDA_CHECK(cudaMalloc((void**)&sc,(size_t)seq*seq*4));
    CUDA_CHECK(cudaMalloc((void**)&ao,xs*4)); CUDA_CHECK(cudaMalloc((void**)&ga,fs*4));
    CUDA_CHECK(cudaMalloc((void**)&ua,fs*4));
    fill_random((float*)x, xs, 0.1f);

    float one=1,z=0, isq=1.0f/sqrtf(128.0f); int gemms=0;
    Timer t; t.start();

    for (int l = 0; l < nl; l++) {
        if (l%8==0) printf("  Layer %d/%d (%.1fs)\n", l, nl, t.ms()/1000);
        rmsnorm((float*)nm,(const float*)x,(const float*)rms_w,d,seq);

        CUBLAS_CHECK(qgemm(cublas,seq,nm,Wq,q,deq,ca));
        CUBLAS_CHECK(qgemm(cublas,seq,nm,Wk,k,deq,ca));
        CUBLAS_CHECK(qgemm(cublas,seq,nm,Wv,v,deq,ca)); gemms+=3;
        rope((float*)q,seq,128,0); rope((float*)k,seq,128,0);

        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_T,seq,seq,d,&isq,q,seq,k,seq,&z,sc,seq));
        gemms++;
        for (int i=0;i<seq;i++) softmax_window((float*)sc+i*seq, i, seq, win);

        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,seq,&one,sc,seq,v,seq,&z,ao,seq));
        gemms++;
        CUBLAS_CHECK(qgemm(cublas,seq,ao,Wo,q,deq,ca)); gemms++;
        CUBLAS_CHECK(cublasSaxpy(cublas,xs,&one,q,1,x,1));

        rmsnorm((float*)nm,(const float*)x,(const float*)rms_w,d,seq);
        CUBLAS_CHECK(qgemm(cublas,seq,nm,Wg,ga,deq,cf));
        CUBLAS_CHECK(qgemm(cublas,seq,nm,Wu,ua,deq,cf)); gemms+=2;
        swiglu((float*)ga,(const float*)ga,(const float*)ua,fs);
        CUBLAS_CHECK(qgemm(cublas,seq,ga,Wd,ao,deq,cd)); gemms++;
        CUBLAS_CHECK(cublasSaxpy(cublas,xs,&one,ao,1,x,1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok = all_finite((const float*)x, xs);
    printf("  %d GEMMs, %.1fs\n", gemms, t.ms()/1000);

    free_qw(Wq);free_qw(Wk);free_qw(Wv);free_qw(Wo);free_qw(Wg);free_qw(Wu);free_qw(Wd);
    cudaFree(rms_w);cudaFree(deq);cudaFree(x);cudaFree(nm);cudaFree(q);cudaFree(k);
    cudaFree(v);cudaFree(sc);cudaFree(ao);cudaFree(ga);cudaFree(ua);
    cublasDestroy(cublas);
    if (!ok) { fprintf(stderr,"  Non-finite\n"); return false; }
    return true;
}

// =====================================================================
// 2. Mixtral 8x7B — Mixture of Experts (FP32)
// =====================================================================
static bool test_mixtral_8x7b() {
    const int d = 4096, d_ff = 14336, nl = 8, seq = 64;
    const int nexp = 8, topk = 2;
    size_t active = 4*(size_t)d*d + topk*(2*(size_t)d*d_ff+(size_t)d_ff*d);
    printf("  %.1fB active/layer, d=%d, %d layers, %d experts top-%d, seq=%d\n",
           active/1e9, d, nl, nexp, topk, seq);

    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas)); srand(8888);

    float *Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wdn,*Wr,*rms_w;
    if (!alloc_weight(&Wq,(size_t)d*d) || !alloc_weight(&Wk,(size_t)d*d) ||
        !alloc_weight(&Wv,(size_t)d*d) || !alloc_weight(&Wo,(size_t)d*d) ||
        !alloc_weight(&Wg,(size_t)d*d_ff) || !alloc_weight(&Wu,(size_t)d*d_ff) ||
        !alloc_weight(&Wdn,(size_t)d_ff*d) || !alloc_weight(&Wr,(size_t)d*nexp))
        { fprintf(stderr,"  OOM\n"); return false; }
    CUDA_CHECK(cudaMalloc((void**)&rms_w,d*4));
    { std::vector<float> o(d,1.0f); cudaMemcpy(rms_w,o.data(),d*4,cudaMemcpyHostToDevice); }

    size_t xs=(size_t)seq*d, fs=(size_t)seq*d_ff;
    float *x,*nm,*q,*k,*v,*sc,*ao;
    float *rl,*ein,*ega,*eua,*eout,*fout;
    CUDA_CHECK(cudaMalloc((void**)&x,xs*4));  CUDA_CHECK(cudaMalloc((void**)&nm,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&q,xs*4));  CUDA_CHECK(cudaMalloc((void**)&k,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&v,xs*4));  CUDA_CHECK(cudaMalloc((void**)&sc,(size_t)seq*seq*4));
    CUDA_CHECK(cudaMalloc((void**)&ao,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&rl,(size_t)seq*nexp*4));
    CUDA_CHECK(cudaMalloc((void**)&ein,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&ega,fs*4)); CUDA_CHECK(cudaMalloc((void**)&eua,fs*4));
    CUDA_CHECK(cudaMalloc((void**)&eout,xs*4)); CUDA_CHECK(cudaMalloc((void**)&fout,xs*4));
    fill_random((float*)x, xs, 0.1f);

    float one=1,z=0, isq=1.0f/sqrtf(128.0f); int gemms=0, edisps=0;
    Timer t; t.start();

    for (int l = 0; l < nl; l++) {
        printf("  Layer %d/%d\n", l, nl);
        rmsnorm((float*)nm,(const float*)x,(const float*)rms_w,d,seq);

        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,d,&one,nm,seq,Wq,d,&z,q,seq));
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,d,&one,nm,seq,Wk,d,&z,k,seq));
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,d,&one,nm,seq,Wv,d,&z,v,seq));
        gemms+=3;
        rope((float*)q,seq,128,0); rope((float*)k,seq,128,0);

        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_T,seq,seq,d,&isq,q,seq,k,seq,&z,sc,seq));
        gemms++;
        for (int i=0;i<seq;i++) softmax_causal((float*)sc+i*seq,i,seq);

        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,seq,&one,sc,seq,v,seq,&z,ao,seq));
        gemms++;
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,d,&one,ao,seq,Wo,d,&z,q,seq));
        gemms++;
        CUBLAS_CHECK(cublasSaxpy(cublas,xs,&one,q,1,x,1));

        // MoE FFN
        rmsnorm((float*)nm,(const float*)x,(const float*)rms_w,d,seq);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,nexp,d,&one,nm,seq,Wr,d,&z,rl,seq));
        gemms++;
        float* rp=(float*)rl;
        for (int i=0;i<seq;i++) softmax_row(rp+i*nexp,nexp);

        std::vector<std::vector<int>> etoks(nexp);
        std::vector<std::vector<float>> ewts(nexp);
        for (int i=0;i<seq;i++) {
            float* r=rp+i*nexp; int e1=0,e2=1;
            if(r[e2]>r[e1]) std::swap(e1,e2);
            for(int e=2;e<nexp;e++){if(r[e]>r[e2]){e2=e;if(r[e2]>r[e1])std::swap(e1,e2);}}
            float ws=r[e1]+r[e2];
            etoks[e1].push_back(i); ewts[e1].push_back(r[e1]/ws);
            etoks[e2].push_back(i); ewts[e2].push_back(r[e2]/ws);
        }

        memset((float*)fout, 0, xs*sizeof(float));
        for (int e = 0; e < nexp; e++) {
            int batch = (int)etoks[e].size(); if (!batch) continue;
            edisps++;
            float* ei=(float*)ein; float* np=(float*)nm;
            for(int i=0;i<batch;i++) memcpy(ei+i*d, np+etoks[e][i]*d, d*sizeof(float));

            CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,batch,d_ff,d,&one,ei,batch,Wg,d,&z,(float*)ega,batch));
            CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,batch,d_ff,d,&one,ei,batch,Wu,d,&z,(float*)eua,batch));
            swiglu((float*)ega,(const float*)ega,(const float*)eua,(size_t)batch*d_ff);
            CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,batch,d,d_ff,&one,(float*)ega,batch,Wdn,d_ff,&z,(float*)eout,batch));
            gemms+=3;

            float* fo=(float*)fout; float* eo=(float*)eout;
            for(int i=0;i<batch;i++){int tok=etoks[e][i];float w=ewts[e][i];
                for(int j=0;j<d;j++) fo[tok*d+j]+=w*eo[i*d+j];}
        }
        CUBLAS_CHECK(cublasSaxpy(cublas,xs,&one,fout,1,x,1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok = all_finite((const float*)x, xs);
    printf("  %d GEMMs, %d expert dispatches, %.1fs\n", gemms, edisps, t.ms()/1000);

    cudaFree(Wq);cudaFree(Wk);cudaFree(Wv);cudaFree(Wo);cudaFree(Wg);cudaFree(Wu);
    cudaFree(Wdn);cudaFree(Wr);cudaFree(rms_w);
    cudaFree(x);cudaFree(nm);cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(sc);cudaFree(ao);
    cudaFree(rl);cudaFree(ein);cudaFree(ega);cudaFree(eua);cudaFree(eout);cudaFree(fout);
    cublasDestroy(cublas);
    if (!ok) { fprintf(stderr,"  Non-finite\n"); return false; }
    return true;
}

// =====================================================================
// 3. DeepSeek-V2 — Multi-head Latent Attention + MoE (INT8)
// =====================================================================
static bool test_deepseek_v2() {
    const int d=5120, d_ff=13824, cq=1536, ckv=512;
    const int nl=8, seq=64, nexp=16, topk=2;
    size_t mla = (size_t)d*cq+(size_t)cq*d+(size_t)d*ckv+2*(size_t)ckv*d+(size_t)d*d;
    size_t moe = (size_t)nexp*(2*(size_t)d*d_ff+(size_t)d_ff*d);
    printf("  %.1fB params, d=%d, c_q=%d, c_kv=%d (%.0fx KV compress)\n",
           (mla+moe)*nl/1e9, d, cq, ckv, (float)(2*d)/ckv);

    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas)); srand(9999);

    QWeight Wdq,Wuq,Wdkv,Wuk,Wuv,Wo,Wg,Wu,Wd;
    if (!alloc_qw(Wdq,d,cq)||!alloc_qw(Wuq,cq,d)||!alloc_qw(Wdkv,d,ckv)||
        !alloc_qw(Wuk,ckv,d)||!alloc_qw(Wuv,ckv,d)||!alloc_qw(Wo,d,d)||
        !alloc_qw(Wg,d,d_ff)||!alloc_qw(Wu,d,d_ff)||!alloc_qw(Wd,d_ff,d))
        { fprintf(stderr,"  OOM\n"); return false; }

    float* Wr; alloc_weight(&Wr,(size_t)d*nexp);
    float* rms_w; CUDA_CHECK(cudaMalloc((void**)&rms_w,d*4));
    { std::vector<float> o(d,1.0f); cudaMemcpy(rms_w,o.data(),d*4,cudaMemcpyHostToDevice); }

    size_t de=(size_t)d*d; float* deq;
    CUDA_CHECK(cudaMalloc((void**)&deq,de*4));
    int cm=d, cf=(int)(de/d), cd=(int)(de/d_ff);

    size_t xs=(size_t)seq*d;
    float *x,*nm,*cqb,*q,*ckvb,*k,*v,*sc,*ao;
    float *rl,*ein,*ega,*eua,*eout,*fout;
    CUDA_CHECK(cudaMalloc((void**)&x,xs*4));    CUDA_CHECK(cudaMalloc((void**)&nm,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&cqb,(size_t)seq*cq*4));
    CUDA_CHECK(cudaMalloc((void**)&q,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&ckvb,(size_t)seq*ckv*4));
    CUDA_CHECK(cudaMalloc((void**)&k,xs*4));    CUDA_CHECK(cudaMalloc((void**)&v,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&sc,(size_t)seq*seq*4));
    CUDA_CHECK(cudaMalloc((void**)&ao,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&rl,(size_t)seq*nexp*4));
    CUDA_CHECK(cudaMalloc((void**)&ein,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&ega,(size_t)seq*d_ff*4));
    CUDA_CHECK(cudaMalloc((void**)&eua,(size_t)seq*d_ff*4));
    CUDA_CHECK(cudaMalloc((void**)&eout,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&fout,xs*4));
    fill_random((float*)x, xs, 0.1f);

    float one=1,z=0, isq=1.0f/sqrtf(128.0f); int gemms=0;
    Timer t; t.start();

    for (int l = 0; l < nl; l++) {
        printf("  Layer %d/%d (%.1fs)\n", l, nl, t.ms()/1000);
        rmsnorm((float*)nm,(const float*)x,(const float*)rms_w,d,seq);

        // MLA: compress Q then expand
        CUBLAS_CHECK(qgemm(cublas,seq,nm,Wdq,cqb,deq,cq));
        CUBLAS_CHECK(qgemm(cublas,seq,cqb,Wuq,q,deq,d)); gemms+=2;
        // MLA: compress KV then expand
        CUBLAS_CHECK(qgemm(cublas,seq,nm,Wdkv,ckvb,deq,ckv));
        CUBLAS_CHECK(qgemm(cublas,seq,ckvb,Wuk,k,deq,d));
        CUBLAS_CHECK(qgemm(cublas,seq,ckvb,Wuv,v,deq,d)); gemms+=3;

        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_T,seq,seq,d,&isq,q,seq,k,seq,&z,sc,seq));
        gemms++;
        for (int i=0;i<seq;i++) softmax_causal((float*)sc+i*seq,i,seq);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,seq,&one,sc,seq,v,seq,&z,ao,seq));
        gemms++;
        CUBLAS_CHECK(qgemm(cublas,seq,ao,Wo,q,deq,cm)); gemms++;
        CUBLAS_CHECK(cublasSaxpy(cublas,xs,&one,q,1,x,1));

        // MoE FFN (same routing as Mixtral but INT8 expert weights)
        rmsnorm((float*)nm,(const float*)x,(const float*)rms_w,d,seq);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,nexp,d,&one,nm,seq,Wr,d,&z,rl,seq));
        gemms++;
        float* rp=(float*)rl;
        for(int i=0;i<seq;i++) softmax_row(rp+i*nexp,nexp);

        std::vector<std::vector<int>> etoks(nexp);
        std::vector<std::vector<float>> ewts(nexp);
        for(int i=0;i<seq;i++){
            float*r=rp+i*nexp;int e1=0,e2=1;
            if(r[e2]>r[e1])std::swap(e1,e2);
            for(int e=2;e<nexp;e++){if(r[e]>r[e2]){e2=e;if(r[e2]>r[e1])std::swap(e1,e2);}}
            float ws=r[e1]+r[e2];
            etoks[e1].push_back(i);ewts[e1].push_back(r[e1]/ws);
            etoks[e2].push_back(i);ewts[e2].push_back(r[e2]/ws);
        }
        memset((float*)fout,0,xs*4);
        for(int e=0;e<nexp;e++){
            int batch=(int)etoks[e].size();if(!batch)continue;
            float*ei=(float*)ein;float*np=(float*)nm;
            for(int i=0;i<batch;i++)memcpy(ei+i*d,np+etoks[e][i]*d,d*4);
            CUBLAS_CHECK(qgemm(cublas,batch,ei,Wg,(float*)ega,deq,cf));
            CUBLAS_CHECK(qgemm(cublas,batch,ei,Wu,(float*)eua,deq,cf));
            swiglu((float*)ega,(const float*)ega,(const float*)eua,(size_t)batch*d_ff);
            CUBLAS_CHECK(qgemm(cublas,batch,(float*)ega,Wd,(float*)eout,deq,cd));
            gemms+=3;
            float*fo=(float*)fout;float*eo=(float*)eout;
            for(int i=0;i<batch;i++){int tok=etoks[e][i];float w=ewts[e][i];
                for(int j=0;j<d;j++)fo[tok*d+j]+=w*eo[i*d+j];}
        }
        CUBLAS_CHECK(cublasSaxpy(cublas,xs,&one,fout,1,x,1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok = all_finite((const float*)x, xs);
    printf("  %d GEMMs, %.1fs, KV cache: %d vs %d full\n", gemms, t.ms()/1000, ckv, 2*d);

    free_qw(Wdq);free_qw(Wuq);free_qw(Wdkv);free_qw(Wuk);free_qw(Wuv);free_qw(Wo);
    free_qw(Wg);free_qw(Wu);free_qw(Wd);
    cudaFree(Wr);cudaFree(rms_w);cudaFree(deq);
    cudaFree(x);cudaFree(nm);cudaFree(cqb);cudaFree(q);cudaFree(ckvb);
    cudaFree(k);cudaFree(v);cudaFree(sc);cudaFree(ao);
    cudaFree(rl);cudaFree(ein);cudaFree(ega);cudaFree(eua);cudaFree(eout);cudaFree(fout);
    cublasDestroy(cublas);
    if(!ok){fprintf(stderr,"  Non-finite\n");return false;}
    return true;
}

// =====================================================================
// 4. Whisper Large v3 — Encoder-Decoder with Cross-Attention (FP32)
// =====================================================================
static bool test_whisper_large() {
    const int d=1280, d_ff=5120, el=8, dl=8, es=200, ds=64;
    printf("  d=%d, d_ff=%d, enc=%d layers (%d pos), dec=%d layers (%d pos)\n",
           d, d_ff, el, es, dl, ds);

    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas)); srand(1234);

    float *Wq,*Wk,*Wv,*Wo,*W1,*W2,*Wqc,*Wkc,*Wvc,*Woc,*lng,*lnb;
    if (!alloc_weight(&Wq,(size_t)d*d)||!alloc_weight(&Wk,(size_t)d*d)||
        !alloc_weight(&Wv,(size_t)d*d)||!alloc_weight(&Wo,(size_t)d*d)||
        !alloc_weight(&W1,(size_t)d*d_ff)||!alloc_weight(&W2,(size_t)d_ff*d)||
        !alloc_weight(&Wqc,(size_t)d*d)||!alloc_weight(&Wkc,(size_t)d*d)||
        !alloc_weight(&Wvc,(size_t)d*d)||!alloc_weight(&Woc,(size_t)d*d))
        { fprintf(stderr,"  OOM\n"); return false; }
    CUDA_CHECK(cudaMalloc((void**)&lng,d*4)); CUDA_CHECK(cudaMalloc((void**)&lnb,d*4));
    { std::vector<float> o(d,1.0f),z(d,0.0f);
      cudaMemcpy(lng,o.data(),d*4,cudaMemcpyHostToDevice);
      cudaMemcpy(lnb,z.data(),d*4,cudaMemcpyHostToDevice); }

    size_t exs=(size_t)es*d, dxs=(size_t)ds*d;
    float *ex,*en,*eq,*ek,*ev,*esc,*eao,*eff,*effo;
    CUDA_CHECK(cudaMalloc((void**)&ex,exs*4));  CUDA_CHECK(cudaMalloc((void**)&en,exs*4));
    CUDA_CHECK(cudaMalloc((void**)&eq,exs*4));  CUDA_CHECK(cudaMalloc((void**)&ek,exs*4));
    CUDA_CHECK(cudaMalloc((void**)&ev,exs*4));  CUDA_CHECK(cudaMalloc((void**)&esc,(size_t)es*es*4));
    CUDA_CHECK(cudaMalloc((void**)&eao,exs*4)); CUDA_CHECK(cudaMalloc((void**)&eff,(size_t)es*d_ff*4));
    CUDA_CHECK(cudaMalloc((void**)&effo,exs*4));

    float *dx,*dn,*dq,*dk,*dv,*dsc,*dao,*dff,*dffo;
    float *cq2,*csc,*cao,*ck2,*cv2;
    CUDA_CHECK(cudaMalloc((void**)&dx,dxs*4));  CUDA_CHECK(cudaMalloc((void**)&dn,dxs*4));
    CUDA_CHECK(cudaMalloc((void**)&dq,dxs*4));  CUDA_CHECK(cudaMalloc((void**)&dk,dxs*4));
    CUDA_CHECK(cudaMalloc((void**)&dv,dxs*4));  CUDA_CHECK(cudaMalloc((void**)&dsc,(size_t)ds*ds*4));
    CUDA_CHECK(cudaMalloc((void**)&dao,dxs*4)); CUDA_CHECK(cudaMalloc((void**)&dff,(size_t)ds*d_ff*4));
    CUDA_CHECK(cudaMalloc((void**)&dffo,dxs*4));
    CUDA_CHECK(cudaMalloc((void**)&cq2,dxs*4));
    CUDA_CHECK(cudaMalloc((void**)&csc,(size_t)ds*es*4)); // rectangular!
    CUDA_CHECK(cudaMalloc((void**)&cao,dxs*4));
    CUDA_CHECK(cudaMalloc((void**)&ck2,exs*4)); CUDA_CHECK(cudaMalloc((void**)&cv2,exs*4));

    fill_random((float*)ex,exs,0.1f); fill_random((float*)dx,dxs,0.1f);
    float one=1,z=0, isq=1.0f/sqrtf(64.0f); int gemms=0;
    Timer t; t.start();

    // Encoder: bidirectional attention
    printf("  Encoder (%d layers, %d positions, bidirectional)...\n", el, es);
    for (int l=0;l<el;l++) {
        layernorm((float*)en,(const float*)ex,(const float*)lng,(const float*)lnb,d,es);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,es,d,d,&one,en,es,Wq,d,&z,eq,es));
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,es,d,d,&one,en,es,Wk,d,&z,ek,es));
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,es,d,d,&one,en,es,Wv,d,&z,ev,es));
        gemms+=3;
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_T,es,es,d,&isq,eq,es,ek,es,&z,esc,es));
        gemms++;
        for(int i=0;i<es;i++) softmax_row((float*)esc+i*es,es); // NO causal mask
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,es,d,es,&one,esc,es,ev,es,&z,eao,es));
        gemms++;
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,es,d,d,&one,eao,es,Wo,d,&z,eq,es));
        gemms++;
        CUBLAS_CHECK(cublasSaxpy(cublas,exs,&one,eq,1,ex,1));

        layernorm((float*)en,(const float*)ex,(const float*)lng,(const float*)lnb,d,es);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,es,d_ff,d,&one,en,es,W1,d,&z,eff,es));
        gemms++;
        gelu_act((float*)eff,(size_t)es*d_ff);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,es,d,d_ff,&one,eff,es,W2,d_ff,&z,effo,es));
        gemms++;
        CUBLAS_CHECK(cublasSaxpy(cublas,exs,&one,effo,1,ex,1));
    }
    int enc_gemms = gemms;

    // Pre-compute cross K,V from encoder output
    CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,es,d,d,&one,ex,es,Wkc,d,&z,ck2,es));
    CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,es,d,d,&one,ex,es,Wvc,d,&z,cv2,es));
    gemms+=2;

    // Decoder: causal self-attn + cross-attn + FFN
    printf("  Decoder (%d layers, %d positions, cross-attn to %d enc positions)...\n", dl, ds, es);
    for (int l=0;l<dl;l++) {
        // Self-attention (causal)
        layernorm((float*)dn,(const float*)dx,(const float*)lng,(const float*)lnb,d,ds);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,ds,d,d,&one,dn,ds,Wq,d,&z,dq,ds));
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,ds,d,d,&one,dn,ds,Wk,d,&z,dk,ds));
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,ds,d,d,&one,dn,ds,Wv,d,&z,dv,ds));
        gemms+=3;
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_T,ds,ds,d,&isq,dq,ds,dk,ds,&z,dsc,ds));
        gemms++;
        for(int i=0;i<ds;i++) softmax_causal((float*)dsc+i*ds,i,ds);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,ds,d,ds,&one,dsc,ds,dv,ds,&z,dao,ds));
        gemms++;
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,ds,d,d,&one,dao,ds,Wo,d,&z,dq,ds));
        gemms++;
        CUBLAS_CHECK(cublasSaxpy(cublas,dxs,&one,dq,1,dx,1));

        // Cross-attention (rectangular: dec_seq × enc_seq)
        layernorm((float*)dn,(const float*)dx,(const float*)lng,(const float*)lnb,d,ds);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,ds,d,d,&one,dn,ds,Wqc,d,&z,cq2,ds));
        gemms++;
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_T,ds,es,d,&isq,cq2,ds,ck2,es,&z,csc,ds));
        gemms++;
        for(int i=0;i<ds;i++) softmax_row((float*)csc+i*es,es); // no causal for cross-attn
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,ds,d,es,&one,csc,ds,cv2,es,&z,cao,ds));
        gemms++;
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,ds,d,d,&one,cao,ds,Woc,d,&z,dq,ds));
        gemms++;
        CUBLAS_CHECK(cublasSaxpy(cublas,dxs,&one,dq,1,dx,1));

        // FFN
        layernorm((float*)dn,(const float*)dx,(const float*)lng,(const float*)lnb,d,ds);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,ds,d_ff,d,&one,dn,ds,W1,d,&z,dff,ds));
        gemms++;
        gelu_act((float*)dff,(size_t)ds*d_ff);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,ds,d,d_ff,&one,dff,ds,W2,d_ff,&z,dffo,ds));
        gemms++;
        CUBLAS_CHECK(cublasSaxpy(cublas,dxs,&one,dffo,1,dx,1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok = all_finite((const float*)ex,exs) && all_finite((const float*)dx,dxs);
    printf("  Enc: %d GEMMs, Dec: %d GEMMs, Total: %d, %.1fs\n",
           enc_gemms, gemms-enc_gemms, gemms, t.ms()/1000);

    cudaFree(Wq);cudaFree(Wk);cudaFree(Wv);cudaFree(Wo);cudaFree(W1);cudaFree(W2);
    cudaFree(Wqc);cudaFree(Wkc);cudaFree(Wvc);cudaFree(Woc);cudaFree(lng);cudaFree(lnb);
    cudaFree(ex);cudaFree(en);cudaFree(eq);cudaFree(ek);cudaFree(ev);cudaFree(esc);
    cudaFree(eao);cudaFree(eff);cudaFree(effo);
    cudaFree(dx);cudaFree(dn);cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dsc);
    cudaFree(dao);cudaFree(dff);cudaFree(dffo);
    cudaFree(cq2);cudaFree(csc);cudaFree(cao);cudaFree(ck2);cudaFree(cv2);
    cublasDestroy(cublas);
    if(!ok){fprintf(stderr,"  Non-finite\n");return false;}
    return true;
}

// =====================================================================
// 5. BERT Large — Bidirectional Encoder (FP32)
// =====================================================================
static bool test_bert_large() {
    const int d=1024, d_ff=4096, nl=12, seq=256;
    size_t ppl = nl*(4*(size_t)d*d+(size_t)d*d_ff+(size_t)d_ff*d);
    printf("  %.2fB params, d=%d, d_ff=%d, %d layers, seq=%d (bidirectional)\n",
           ppl/1e9, d, d_ff, nl, seq);

    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas)); srand(5555);

    float *Wq,*Wk,*Wv,*Wo,*W1,*W2,*lng,*lnb;
    if (!alloc_weight(&Wq,(size_t)d*d)||!alloc_weight(&Wk,(size_t)d*d)||
        !alloc_weight(&Wv,(size_t)d*d)||!alloc_weight(&Wo,(size_t)d*d)||
        !alloc_weight(&W1,(size_t)d*d_ff)||!alloc_weight(&W2,(size_t)d_ff*d))
        { fprintf(stderr,"  OOM\n"); return false; }
    CUDA_CHECK(cudaMalloc((void**)&lng,d*4)); CUDA_CHECK(cudaMalloc((void**)&lnb,d*4));
    { std::vector<float> o(d,1.0f),z(d,0.0f);
      cudaMemcpy(lng,o.data(),d*4,cudaMemcpyHostToDevice);
      cudaMemcpy(lnb,z.data(),d*4,cudaMemcpyHostToDevice); }

    size_t xs=(size_t)seq*d;
    float *x,*nm,*q,*k,*v,*sc,*ao,*ff,*ffo;
    CUDA_CHECK(cudaMalloc((void**)&x,xs*4));  CUDA_CHECK(cudaMalloc((void**)&nm,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&q,xs*4));  CUDA_CHECK(cudaMalloc((void**)&k,xs*4));
    CUDA_CHECK(cudaMalloc((void**)&v,xs*4));  CUDA_CHECK(cudaMalloc((void**)&sc,(size_t)seq*seq*4));
    CUDA_CHECK(cudaMalloc((void**)&ao,xs*4)); CUDA_CHECK(cudaMalloc((void**)&ff,(size_t)seq*d_ff*4));
    CUDA_CHECK(cudaMalloc((void**)&ffo,xs*4));
    fill_random((float*)x, xs, 0.1f);

    float one=1,z2=0, isq=1.0f/sqrtf(64.0f); int gemms=0;
    Timer t; t.start();

    for (int l=0;l<nl;l++) {
        layernorm((float*)nm,(const float*)x,(const float*)lng,(const float*)lnb,d,seq);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,d,&one,nm,seq,Wq,d,&z2,q,seq));
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,d,&one,nm,seq,Wk,d,&z2,k,seq));
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,d,&one,nm,seq,Wv,d,&z2,v,seq));
        gemms+=3;
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_T,seq,seq,d,&isq,q,seq,k,seq,&z2,sc,seq));
        gemms++;
        // BIDIRECTIONAL: no causal mask
        for(int i=0;i<seq;i++) softmax_row((float*)sc+i*seq,seq);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,seq,&one,sc,seq,v,seq,&z2,ao,seq));
        gemms++;
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,d,&one,ao,seq,Wo,d,&z2,q,seq));
        gemms++;
        CUBLAS_CHECK(cublasSaxpy(cublas,xs,&one,q,1,x,1));

        layernorm((float*)nm,(const float*)x,(const float*)lng,(const float*)lnb,d,seq);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d_ff,d,&one,nm,seq,W1,d,&z2,ff,seq));
        gemms++;
        gelu_act((float*)ff,(size_t)seq*d_ff);
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,seq,d,d_ff,&one,ff,seq,W2,d_ff,&z2,ffo,seq));
        gemms++;
        CUBLAS_CHECK(cublasSaxpy(cublas,xs,&one,ffo,1,x,1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float cls_rms=0;
    for(int i=0;i<d;i++) cls_rms+=((float*)x)[i]*((float*)x)[i];
    cls_rms=sqrtf(cls_rms/d);
    bool ok = all_finite((const float*)x, xs);
    printf("  %d GEMMs, %.1fs, [CLS] RMS=%.6f\n", gemms, t.ms()/1000, cls_rms);

    cudaFree(Wq);cudaFree(Wk);cudaFree(Wv);cudaFree(Wo);cudaFree(W1);cudaFree(W2);
    cudaFree(lng);cudaFree(lnb);
    cudaFree(x);cudaFree(nm);cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(sc);
    cudaFree(ao);cudaFree(ff);cudaFree(ffo);
    cublasDestroy(cublas);
    if(!ok){fprintf(stderr,"  Non-finite\n");return false;}
    return true;
}

// =====================================================================
// Main
// =====================================================================
int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        printf("Device: %s (%.1f MB)\n\n", prop.name, prop.totalGlobalMem/(1024.0*1024));

    printf("==============================\n");
    printf("Multi-Model Architecture Tests\n");
    printf("==============================\n\n");

    int pass=0, fail=0;
    struct { const char* name; bool(*fn)(); } tests[] = {
        {"Mistral 7B (sliding window, INT8)", test_mistral_7b},
        {"Mixtral 8x7B (MoE top-2, FP32)", test_mixtral_8x7b},
        {"DeepSeek-V2 (MLA + MoE, INT8)", test_deepseek_v2},
        {"Whisper Large v3 (enc-dec)", test_whisper_large},
        {"BERT Large (bidirectional)", test_bert_large},
    };

    for (auto& t : tests) {
        printf("--- %s ---\n", t.name);
        Timer timer; timer.start();
        if (t.fn()) { pass++; printf("  PASS (%.1fs)\n\n", timer.ms()/1000); }
        else { fail++; printf("  FAIL\n\n"); }
    }

    printf("=== %d/%d passed ===\n", pass, pass+fail);
    return fail ? 1 : 0;
}
