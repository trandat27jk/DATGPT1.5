# DATGPT 1.5: A GPT Model with Normalized Transformer Architecture
![DATGPT 1 5](https://github.com/user-attachments/assets/b9c92bdf-e851-4041-a08c-abbfc7d9fa76)

DATGPT 1.5 is a mini language model built based on the nGPT: Normalized Transformer with Representation Learning on the Hypersphere paper https://arxiv.org/abs/2410.01131. This model leverages advanced normalization techniques and hypersphere-based representation learning to enhance performance across diverse NLP tasks. It is a central component of my graduation thesis, demonstrating the application of innovative transformer architectures.

# Features
**Normalized Transformer Architecture**: Implements hypersphere-based representation learning for improved generalization.

**Faster Convergence**: Testing confirms the model converges 4-20x faster as claimed in the original paper.

**Customizable Configuration**: Easily modify model parameters like the number of layers, heads, and embedding dimensions via config.yaml.

**GPT-2 Tokenizer**: Uses a proven and efficient tokenizer for text processing.

**Dataset Options**: Currently uses The Verdict dataset, with plans to switch to the FineWeb dataset for larger-scale training.
Current Configuration
The model is initialized with the following settings (modifiable via config.yaml):

Layers: 12  
Heads: 8   
Embedding Dimensions: 768  
Tokenizer: GPT-2  
# How to Train
To train the model, run the following command:
```
torchrun --standalone --nproc_per_node=1 main.py
```
#Customization
Modify the model architecture or tokenizer by updating the config.yaml file with your desired settings:
```
mininGPT:  
  n_layers: <number_of_layers>  
  n_heads: <number_of_heads>  
  n_embd: <embedding_dimension>
```
# Future Improvements
+) FineWeb Dataset: Transitioning to training on the large-scale FineWeb dataset.  
+) Gradient Accumulation: Plan to add gradient accumulation for better optimization and handling larger batch sizes.  
+) Fine-Tuning: Fine-tune on domain-specific datasets for specialized tasks.  
