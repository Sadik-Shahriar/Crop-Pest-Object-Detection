======================================================================
PHASE 1: Auditing labels across all splits
======================================================================
   train: 27 conflicts
   valid: 4 conflicts
    test: 0 conflicts

Total: 31 conflicts / ~13K files (0.24%)
  Safe to fix (single-class files): 3
  Needs review (mixed-class files): 28

split	prefix	filename	wrong_name	correct_name	safe_to_fix
0	train	ants-1	ants-1-_png.rf.0b5a6cf9f49f735808979cd685d8ce4...	Ants	Bees	True
1	train	ants-1	ants-1-_png.rf.809de1265776e6576f1633d8656dda4...	Ants	Bees	True
2	train	ants-1	ants-1-_png.rf.e4e6047152ea7ab2d012b00013d1029...	Ants	Bees	True
3	train	beetle-252	beetle-252-_jpg.rf.0f36b554b30566e4f6e7c80efa2...	Wasps	Beetles	False
4	train	beetle-252	beetle-252-_jpg.rf.ccbfbfde3acead1d7ffd78ec182...	Wasps	Beetles	False
5	train	beetle-252	beetle-252-_jpg.rf.d56ee69a87b00df58f94451cb54...	Wasps	Beetles	False
6	train	catterpillar-235	catterpillar-235-_jpg.rf.a3148298d5b74d9c68c1f...	Bees	Caterpillars	False
7	train	catterpillar-235	catterpillar-235-_jpg.rf.ce71e419f4ba17604cfbe...	Bees	Caterpillars	False
8	train	catterpillar-235	catterpillar-235-_jpg.rf.de8c9bbb2b6560baaff0b...	Bees	Caterpillars	False
9	train	catterpillar-419	catterpillar-419-_jpg.rf.362550b1fdd05f1eaf60b...	Caterpillars	Wasps	False
10	train	catterpillar-419	catterpillar-419-_jpg.rf.ad7d3139c5a7331e7cdb2...	Caterpillars	Wasps	False
11	train	catterpillar-419	catterpillar-419-_jpg.rf.ec5bae23c665354e10dcb...	Caterpillars	Wasps	False
12	train	earthworms-231	earthworms-231-_jpg.rf.425a4b03c8bc0943295e115...	Earthworms	Ants	False
13	train	earthworms-231	earthworms-231-_jpg.rf.a73872d528699244f0ce7b3...	Earthworms	Ants	False
14	train	earthworms-231	earthworms-231-_jpg.rf.c017f567abfecc2d71ac0f7...	Earthworms	Ants	False
15	train	earwig-299	earwig-299-_jpg.rf.2b1cce2ec15f412f0c0718927bd...	Wasps	Earwigs	False
16	train	earwig-299	earwig-299-_jpg.rf.308e9fbf6e2aa2bf50f150d8589...	Wasps	Earwigs	False
17	train	earwig-299	earwig-299-_jpg.rf.ae7dbf893bed418c156e3078c7b...	Wasps	Earwigs	False
18	train	earwig-345	earwig-345-_jpg.rf.0c6ab0bc8fab05afee832ac1463...	Earthworms	Earwigs	False
19	train	earwig-345	earwig-345-_jpg.rf.6355c9364b7a932299a9d969881...	Earthworms	Earwigs	False
20	train	earwig-345	earwig-345-_jpg.rf.ccf204d02e78723c9b050aa81e8...	Earthworms	Earwigs	False
21	train	slug-56	slug-56-_jpg.rf.48920b05e2caccf85f64de9bfa93fd...	Snails	Slugs	False
22	train	slug-56	slug-56-_jpg.rf.9b695747bd265db24ca56529896030...	Snails	Slugs	False
23	train	slug-56	slug-56-_jpg.rf.cf5bd98e2b0535d72eae5cfdbffa9a...	Snails	Slugs	False
24	train	slug-57	slug-57-_jpg.rf.38c3e4e91bac3da910ac8dd676645f...	Snails	Slugs	False
25	train	slug-57	slug-57-_jpg.rf.a3ca3958482a694b6d984e774ab04b...	Snails	Slugs	False
26	train	slug-57	slug-57-_jpg.rf.b3a5596143c4a813e114ce853b631d...	Snails	Slugs	False
27	valid	ants-258	ants-258-_jpg.rf.c8043afecbd090379facd32c3c907...	Beetles	Ants	False
28	valid	catterpillar-173	catterpillar-173-_jpg.rf.1a2aa01f8cd54c2a9361b...	Snails	Caterpillars	False
29	valid	earthworms-132	earthworms-132-_jpg.rf.edcb9ed1f6204188616440b...	Earthworms	Slugs	False
30	valid	earthworms-46	earthworms-46-_jpg.rf.51ec4fd8cb02197ff2940101...	Earthworms	Slugs	False

Conflict summary:
    6 files: Snails → Slugs
    3 files: Ants → Bees
    3 files: Wasps → Beetles
    3 files: Bees → Caterpillars
    3 files: Caterpillars → Wasps
    3 files: Earthworms → Ants
    3 files: Wasps → Earwigs
    3 files: Earthworms → Earwigs
    2 files: Earthworms → Slugs
    1 files: Beetles → Ants
    1 files: Snails → Caterpillars

APPLY_FIXES = False — no changes made.
Set APPLY_FIXES = True at top of cell to fix conflicting labels.