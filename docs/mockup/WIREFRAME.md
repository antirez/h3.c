# Wireframe di riferimento — web UI h3.c

Materiale approvato durante la progettazione della web UI.

Schermata 1 — Simple (default):

```
+------------------------------------------------------------------------------+
|  h3.c Studio                       GPU: NVIDIA GB10 - 121 GiB - CUDA 13.0  *  |
+------------------------------------------------------------------------------+
| [ Simple ] [ Advanced ]                                     Queue: 1 running  |
+------------------------------------------+-----------------------------------+
| PROMPT                                   | QUEUE                             |
| +--------------------------------------+ | +-------------------------------+ |
| | A red fox walks through fresh snow.  | | | > #12 "fox in snow"   running | |
| +--------------------------------------+ | |   denoise     7/20  [####...] | |
|                                          | |   03:41 elapsed          [x]  | |
| DURATION   o-------*----------  4.46 s   | +-------------------------------+ |
|            107 frames (aligned 5+17n)    | |   #11 "surfer"    queued  [x] | |
|                                          | +-------------------------------+ |
| FORMAT   [16:9] [9:16] [1:1] [4:3] [3:4] | |   #10 "cube"   done 00:41 > v | |
| SIZE     ( ) 256  (*) 512  ( ) 768       | +-------------------------------+ |
|          512 x 512 - 0.26 / 1.03 MP      |                                   |
|                                          | PREVIEW                           |
| QUALITY  o-------*----------o            | +-------------------------------+ |
|          Draft  Balanced  Reference      | |                               | |
|          steps 20 - layers 45 - reuse 2  | |        [ video player ]       | |
|                                          | |                               | |
| FIRST FRAME         LAST FRAME           | +-------------------------------+ |
| +-------------+     +-------------+      | #10 - 512x512 - 22f - seed 42     |
| |  drop image |     |  drop image |      | [ Reuse settings ]  [ Download ]  |
| +-------------+     +-------------+      |                                   |
|                                          |                                   |
| SEED  [ 42 ]  [ random ]                 |                                   |
|                                          |                                   |
|          [    Generate video    ]        |                                   |
+------------------------------------------+-----------------------------------+
```

Schermata 2 — Advanced (stessa colonna destra, pannello sinistro a sezioni):

```
+------------------------------------------+
| [ Simple ] [ Advanced ]                  |
+------------------------------------------+
| v OUTPUT                                 |
|   width  [ 512 ]  height [ 512 ]  (x32)  |
|   internal canvas  [x] custom            |
|     render-width [384] render-height[384]|
|   output file [ outputs/fox.mp4 ]        |
|   [ ] no mp4 (-o '')                     |
|   [ ] write frames  frames-dir [ ...   ] |
+------------------------------------------+
| v DURATION                               |
|   (*) seconds [ 4.5 ]  ( ) frames [107]  |
|   -> aligned 107 frames = 4.458 s @24fps |
+------------------------------------------+
| v SAMPLER                                |
|   steps      [ 20 ]        (2..1000)     |
|   layers     [ 45 ]        (35..50)      |
|   (*) reuse       [ 2 ]    (1..3)        |
|   ( ) core-reuse  [ 4 ]    (1..6)        |
|       ^ mutually exclusive               |
|   [ ] token-reduction                    |
+------------------------------------------+
| v MEMORY / BACKEND                       |
|   [ ] ssd-streaming   27.1 GB -> 1.6 GB, |
|                       slower             |
|   [ ] use-int8-row-fc2   (no-op on CUDA) |
|   [ ] use-reference-rope                 |
+------------------------------------------+
| > PARITY / DEBUG FLAGS            (10)   |
|   collapsed: --use-slower-*              |
|   [ ] profile                            |
+------------------------------------------+
| > REFERENCES                       (3)   |
+------------------------------------------+
|          [    Generate video    ]        |
+------------------------------------------+
```

Schermata 3 — References (sezione espansa, lista ordinata, max 12):

```
+--------------------------------------------------------------+
| REFERENCES (Ref2VA)            order is significant           |
| ref-image-size:  (*) match   ( ) max                          |
+--------------------------------------------------------------+
| 1 [img]   fox.png                     720x720      [^][v][x]  |
| 2 [vid]   clip.mp4    audio: keep (--ref-video)    [^][v][x]  |
| 3 [vid]   silent.mp4  audio: drop (--ref-silent)   [^][v][x]  |
| 4 [vid+a] scene.mp4 + music.wav                    [^][v][x]  |
| 5 [aud]   music.wav                   6.2 s        [^][v][x]  |
+--------------------------------------------------------------+
| [ + image ] [ + video ] [ + video+audio ] [ + audio ]         |
| rules: audio 2-15 s, max 3 audio, total <= 15 s, audio only   |
|        alongside an image or a video reference                |
+--------------------------------------------------------------+
```
