# Shap-E: Generating Conditional 3D Implicit Functions

This repository provides the official code and model release for **Shap-E**, a model for generating 3D objects conditioned on text or images.

**Original Repository**: [https://github.com/openai/shap-e](https://github.com/openai/shap-e)

## Overview

Shap-E allows for the generation of 3D objects based on text prompts or image inputs, enabling the creation of various creative 3D shapes like:
- A chair that looks like an avocado
- An airplane that looks like a banana
- A birthday cupcake
- Ube ice cream cone
- A penguin

These are just examples of what the model can create based on given prompts.

## Installation

To install Shap-E, clone the repository and install it in editable mode with pip:

```bash
git clone https://github.com/openai/shap-e.git
cd shap-e
pip install -e .
```

### Requirements

- Python 3.x
- Blender (version 3.3.1 or higher) - Required only if using the `encode_model.ipynb` notebook

Set the `BLENDER_PATH` environment variable to the path of the Blender executable if using the encoding functionality.

## Resources

- **Repository**: [OpenAI Shap-E on GitHub](https://github.com/openai/shap-e)

