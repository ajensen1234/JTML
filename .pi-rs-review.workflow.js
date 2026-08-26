const meta = { name: 'jtml_rs_code_review', description: 'CE-style multi-persona review of rust/direct-rs with rust-skills rubric' }
const ROOT = '/home/ajj/repo/uf/JTML'
const SCOPE = [
'REPO: ' + ROOT + ' (Jujutsu VCS. You are STRICTLY READ-ONLY: no edits, no jj/git mutations.)',
'CRATE: rust/direct-rs — Rust port of the DIRECT global optimizer (Jones 1993) for 2D-3D knee-implant registration, linked into a C++/Qt/CUDA host via cxx (crate-type staticlib+cdylib).',
'INTENT: correctness vs the reference algorithm (POH convex-hull selection, trisection, volume partition), bit-exact determinism between identical runs, batched cost evaluation, full metamorphic property suite (affine equivariance, axis permutation, prefix-of-budget, lattice membership). Two expensive tests are deliberately #[ignore]d with reason strings.',
'SCOPE — read every listed file completely before reporting:',
ROOT + '/rust/direct-rs/src/lib.rs',
ROOT + '/rust/direct-rs/src/direct_optimizer.rs',
ROOT + '/rust/direct-rs/src/direct_data_storage.rs',
ROOT + '/rust/direct-rs/src/bench.rs',
ROOT + '/rust/direct-rs/src/properties.rs',
ROOT + '/rust/direct-rs/src/test_support.rs',
ROOT + '/rust/direct-rs/src/test.rs',
ROOT + '/rust/direct-rs/src/utils.rs',
ROOT + '/rust/direct-rs/Cargo.toml',
ROOT + '/rust/Cargo.toml',
].join('\n')
const MANDATE = [
'MANDATORY RUBRIC: first read /home/ajj/.pi/agent/skills/rust-skills/SKILL.md, then read the rule files listed below from /home/ajj/.pi/agent/skills/rust-skills/rules/ and apply them explicitly as your judging standard. Every finding should cite the rule id it violates (e.g. "[obs-tracing-over-log]") when applicable.',
'REPORTING CONTRACT: return ONLY through the structured_output mechanism. At most your 10 STRONGEST findings, ranked by severity. Every finding must be actionable with a concrete fix; cite exact file and current line numbers you personally verified by reading the file. Do NOT report anything a configured linter/clippy would already flag mechanically, pure formatting, or vague suggestions ("consider improving"). If surrounding code shows an apparent issue is actually handled, drop it.',
'Severity scale: P0 = breakage/data-corruption possible in production use; P1 = likely-hit defect breaking contract; P2 = meaningful edge-case/perf/maintainability downside; P3 = minor polish.',
'Confidence anchors (integers only): 100 certain, 75 likely, 50 plausible, 25 speculative. Sub-75 findings will be gated, so only include what you stand behind.'
].join('\n')
function rules(ids){ return ids.map(function(id){ return id + '.md' }).join(', ') }
const REVIEWERS = [
{ key:'correctness', agent:'ce-correctness-reviewer', extra:'LENS: logic errors, off-by-one in trisection/depth accounting, float-order sensitivity, error propagation across the cxx boundary, state bugs in the optimizer loop, intent-vs-implementation mismatches.', r:['err-result-over-panic','err-custom-type','err-no-unwrap-prod','err-source-chain','num-float-compare','num-overflow-explicit','pat-exhaustive-enum','anti-panic-expected'] },
{ key:'testing', agent:'ce-testing-reviewer', extra:'LENS: coverage gaps over the public API and ffi surface, weak assertions, missing edge cases (zero budget, empty POH, degenerate ranges, NaN/+Inf costs, repeated run() calls on same instance), brittleness of threshold-style assertions.', r:['test-descriptive-names','test-fixture-raii','test-proptest-properties','test-should-panic','test-criterion-bench','test-integration-dir'] },
{ key:'maintainability', agent:'ce-maintainability-reviewer', extra:'LENS: premature abstraction, dead code remnants, coupling between data-storage and optimizer internals, naming that obscures DIRECT terminology, API-shape issues on the public surface (new/run/best/RunOutcome), duplicate logic.', r:['api-builder-pattern','api-newtype-safety','api-must-use','api-common-traits','api-from-not-into','type-newtype-ids','type-repr-transparent','name-as-free','name-to-expensive','proj-pub-crate-internal','trait-dyn-vs-generic'] },
{ key:'performance', agent:'ce-performance-reviewer', extra:'LENS: allocation churn in the hot loop, recompute of box size/powf, BTreeMap vs alternative structures for POH candidate extraction, release-profile settings for a library linked into a C++ host, batch-boundary overhead at the FFI seam.', r:['mem-with-capacity','mem-reuse-collections','opt-lto-release','opt-codegen-units','opt-inline-small','perf-release-profile','coll-map-choice','perf-black-box-bench','anti-format-hot-path'] },
{ key:'adversarial', agent:'ce-adversarial-reviewer', extra:'LENS: actively construct inputs that break the implementation — huge/tiny budgets, ranges of zero or NaN, costs returning +/-Infinity mixed with NaN, POH sets of size 0 or 1 mid-run, calls to run() twice on one optimizer, downstream C++ handing a hostile CppCost, integer u64 next_box_id exhaustion, budget arithmetic overflow.', r:['unsafe-safety-comment','unsafe-minimize-scope','num-nonzero','num-saturating-clamp','num-cast-try-from','conc-atomic-ordering'] },
{ key:'standards', agent:'ce-project-standards-reviewer', extra:'LENS: compliance with the repo AGENTS.md conventions (pixi tasks, jj workflow, layered-lib gotchas), the crate-owned workspace clippy lint policy ([lints] workspace = true; allow_attributes banned in favor of #[expect(.., reason)]), missing docs on public items, printing to stdout from library code, Cargo metadata quality.', r:['lint-deny-correctness','lint-workspace-lints','lint-rustfmt-check','doc-all-public','doc-errors-section','obs-tracing-over-log','obs-library-facade','proj-msrv-declare','proj-lib-main-split'] },
]
function mkTask(r){
  return [SCOPE, '', MANDATE, '', 'RULE FILES TO APPLY: ' + rules(r.r), '', 'PERSONA LENS: ' + r.extra].join('\n')
}
const SCHEMA = {
  type:'object', additionalProperties:false,
  required:['reviewer','findings','residual_risks','testing_gaps'],
  properties:{
    reviewer:{type:'string'},
    findings:{type:'array', maxItems:10, items:{
      type:'object', additionalProperties:false,
      required:['title','severity','file','line','confidence','autofix_class','pre_existing'],
      properties:{
        title:{type:'string'},
        severity:{type:'string', enum:['P0','P1','P2','P3']},
        file:{type:'string'},
        line:{type:'integer'},
        confidence:{type:'integer', enum:[0,25,50,75,100]},
        autofix_class:{type:'string', enum:['safe_auto','gated_auto','manual','advisory']},
        pre_existing:{type:'boolean'},
        requires_verification:{type:'boolean'},
        suggested_fix:{type:'string'},
        why_it_matters:{type:'string'},
        rule_id:{type:'string'}
      }
    }},
    residual_risks:{type:'array', items:{type:'string'}},
    testing_gaps:{type:'array', items:{type:'string'}}
  }
}
const results = await runs.all(REVIEWERS.map(function(r){
  return { key:r.key, agent:r.agent, task:mkTask(r), outputSchema:SCHEMA }
}))
return { reviews: results }
