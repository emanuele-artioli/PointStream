# Card: human-directed VISOR proposal refinement

## Goal, scope and stop

Import the genuine18-frame review unchanged, address the foreground-definition mismatch, and prepare better source-only proposals on the same18frames. No codec output, quality metric, confirmation result, training or reference approval is produced. Original statuses9uncertain/9rejected remain unchanged; rough polygons are guidance only. The five original blind examples deliberately hid proposals and are not detector-failure evidence.

The user wants people/hands and objects moved or directly manipulated, including a tap being used; unrelated textiles and whole static sink/oven/stove regions are excluded. Held objects remain relevant while momentarily stationary. A text segmentation alone does not prove physical motion or interaction. The source-specific target policy is frozen in `docs/experiments/visor-action-policy-v2-20261115.json` before model outcomes; historical metric targets are unchanged.

Keep filtered official hand/task-object proposals. Add SAM person/hand instances to recover actors; add SAM task-object instances only when the allowed official object label is absent. Actor prompts are text-only; rejected rough polygons provide an optional task-object bounding box, never accepted pixels. Preserve every raw model instance, score, source identity, conversion and missing case. Revised proposals are visible in all18frames; original blind drafts remain a separate tier, not independent final gold masks. Every final reference still requires human approval.

Stop on failed identity/kernel/output/cost checks, exhausted allocation, completed decision or **2026-11-15T11:07:34Z**. No unknown job is replayed. No GPU retry or deadline extension is authorized here.

## Stage and allocation

The user's existing12-hour automatic budget authority covers this stage. Total ceiling: **3,960CPU seconds /960GPU seconds**: human import/audit600CPU; official filtering900CPU; SAM campaign960CPU/GPU; preparation/publication900CPU; failure allowance600CPU, no GPU retry. All stages end within45minutes and before the authority expires.

One fleet campaign:3representative smoke frames(P35native184,P12native238,P11native241), then all18native images. Smoke≤180s; full≤600s; two≤60s validators;60s lifecycle margin;960s wall cap. One initialized pinned SAM3.1 model, independent one-frame sessions per prompt, no propagation between unrelated frames. Forecast full cost from real smoke with1.5margin; stop if it exceeds600s. Reuse recorded valid Ada/A6000 qualification, while verifying exact model, CUDA/bfloat16/native-flash execution on this actual path. One CPU,14GiB GPU cap plus8GiB free margin; shared timing has no throughput claim. Files/intermediates go to local SSD scratch, publication as one archive.

Input archiveSHA3be31039083b35d3c319f9e9e671dd45d7732114d3c2b790aedf3eb53735b86a contains18original PNGs plus manifest. ModelSHA0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6, installedSAMcode2345a4ad109ac29c569da749c91d84f10dc08c40, packed runtime32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661. No downloads or changed provider. If this path fails, retain the explicit incomplete official-filter proposals and the failure; do not manufacture masks or approvals.
