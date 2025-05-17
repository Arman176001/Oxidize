# Oxidize – Python-to-Rust Code Translator Using Transformers

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Status](https://img.shields.io/badge/Status-Under_Development-orange)


**Oxidize** is an open-source research project that aims to bridge the gap between high-level Python code and low-level Rust performance using deep learning and AST-based techniques.  
Our goal: make it easier for developers and teams to **automatically translate Python code into idiomatic, safe, and high-performance Rust**.

---

## Key Features

- **Custom Encoder-Decoder Transformer** with Multi-Head Latent Attention & Rotary Embeddings  
- **AST-based rule fallback system** for syntax-valid Rust code  
- Semi-supervised dataset builder to align Python and Rust code pairs  
- Researched with foundations from *Attention Is All You Need* and *DeepSeek Coder*  
- Modular, GPU-ready architecture – open for contributors to train, refine, or extend

---

## Why This Project?

Python is great for quick prototyping, but Rust is unbeatable in performance and safety. Instead of rewriting entire codebases manually, **Oxidize aims to automate the migration process**, freeing developers to focus on logic and design.

This project started as a solo research experiment and is now open to the community to:
- Improve model performance
- Expand dataset collection
- Contribute to AST converters
- Deploy for real-world use cases

---
## 🚀 Emergence of the Idea

The idea for this project emerged from the groundbreaking work done by **DeepMind's AlphaCode**, which proved that large language models can solve programming problems with human-level proficiency. 

Inspired by this, the goal was to build a similar **code-generation model**, but with enhanced efficiency and scalability. To do this, we took the following approach:
- **Training Strategy**: Adopted from AlphaCode—using AdamW, gradient clipping, learning rate warm-up and cosine decay.
- **Architecture Upgrade**: Instead of the standard transformer used by AlphaCode, we implemented architectural innovations from **DeepSeek-V3**, particularly improvements in the attention mechanism and feed-forward layers.

This fusion allows for a high-performance model tailored for structured problem-solving and competitive programming tasks.

---

## Project Architecture

> ![Architecture Diagram](docs/architecture_diagram.png)

- `encoder_decoder/`: Transformer model with caching, rotary embeddings, and token-wise attention  
- `ast_parser/`: Python and Rust AST traversal and rule-based translator  
- `dataset/`: Collects, preprocesses, and aligns Python-Rust code pairs  
- `training/`: Model training, evaluation, and logging  

---

## Repository Structure

oxidize/
├── docs/
├── src/
├── notebooks/
├── examples/
├── tests/
├── README.md
├── requirements.txt
└── LICENSE


---
## 📦 Installation

```bash
git clone https://github.com/yourusername/edtransformer.git
cd edtransformer
pip install -r requirements.txt
```

---

## Current Status

✅ Model architecture and training Loop implemented  
✅ Dataset collection initiated (~200+ pairs)  
❌ Full model training pending (GPU access needed)  
🔜 Exploring evaluation and Rust idiomatic correctness scoring

---

## 👐 Contributing

We welcome any and all contributors! Here’s how you can help:

- Improve the AST-to-Rust converter  
- Add better dataset alignment logic  
- Add pre/post-processing layers for inference  
- Write evaluation pipelines  
- Help with training the model if you have GPU access

Just fork it, open a PR, or drop issues/discussions if you have ideas. Let's build it together 🚀

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for more details.

---

## Acknowledgements

- [Competition-Level Code Generation with AlphaCode](https://arxiv.org/pdf/2203.07814)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [DeepSeek V3](https://arxiv.org/pdf/2412.19437)
- [Rust LeetCode Source](https://github.com/warycat/rustgym)
- [Python LeetCode Source](https://github.com/Garvit244/Leetcode)

---

## ❤️ Maintained by [Arman Chaudhary](https://github.com/Arman176001)

This project was originally built as part of a custom research initiative. If you’re a researcher, developer, or Rust/Python enthusiast — your ideas, code, and support are welcome!

