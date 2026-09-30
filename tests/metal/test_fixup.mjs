#!/usr/bin/env node
// test_fixup.mjs — Apply fixMetal() to all .metal files, write to temp dir, then compile-check
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'fs';
import { join, basename } from 'path';
import { execSync, spawnSync } from 'child_process';
import { tmpdir } from 'os';
import { fixMetal } from '../../site/metalfixup.js';

const srcDir = process.argv[2] || `${process.env.HOME}/Downloads/metal_shaders`;
const tmp = `${tmpdir()}/metal_fixup_test_${Date.now()}`;
mkdirSync(tmp, { recursive: true });

const files = readdirSync(srcDir).filter(f => f.endsWith('.metal'));
let warnings = 0;
for (const name of files) {
  const src = readFileSync(join(srcDir, name), 'utf8');
  const { fixed, warnings: w } = fixMetal(src);
  writeFileSync(join(tmp, name), fixed, 'utf8');
  if (w.length) warnings++;
}
console.log(`Processed ${files.length} files → ${tmp}`);

const r = spawnSync('bash', [
  join(import.meta.dirname, 'compile_check.sh'), tmp
], { stdio: 'inherit' });
process.exit(r.status ?? 0);
