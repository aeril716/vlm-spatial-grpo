# Image source and license

The images in `images/` (and the questions, choices and correct answers in `items.json`) come from the
**SAT (Spatial Aptitude Training)** dataset, train split, `array/SAT` on Hugging Face
(https://huggingface.co/datasets/array/SAT, revision `bda5dde942d7f7b41bff7935f086ed9a9e348ae3`).
They were resized to at most 768 px on the longer side and re-encoded as JPEG quality 85.

The dataset card declares `license: mit`. It names no copyright holder. Authors, per the card's citation:
Arijit Ray, Jiafei Duan, Ellis Brown, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani,
Aniruddha Kembhavi, Bryan A. Plummer, Ranjay Krishna, Kuo-Hao Zeng, Kate Saenko.
"SAT: Dynamic Spatial Aptitude Training for Multimodal Language Models", arXiv:2412.07755 (2025).

The model answers in `items.json` were generated in this repo by Qwen2.5-VL-7B-Instruct (Part G).

## MIT License (as declared for the dataset)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
