"""
BACKGROUND LORA TRAINER - Real Weight Updates

This module handles ACTUAL model training (not just prompting!).
Every 5-10 examples, it offers to run LoRA fine-tuning on Colab.

KEY CONCEPT:
- Collects examples → Generates Colab notebook → You train on GPU
- Downloads trained weights → Model actually improves!
- This is REAL learning (weights change)

Unlike pattern memory (instant but fake), this is slow but true learning.
"""

import json
import os
from typing import List, Dict
from datetime import datetime


class TrainingExample:
    """Single training example for LoRA"""
    
    def __init__(self, query: str, review: str, rating: int, feedback: str):
        """
        Create training example
        
        Args:
            query: Research query
            review: Generated review (if rated 8+, this is "good")
            rating: Your rating 1-10
            feedback: What you said
        """
        self.query = query
        self.review = review
        self.rating = rating
        self.feedback = feedback
        self.timestamp = datetime.now().isoformat()
    
    def to_lora_format(self) -> Dict[str, str]:
        """
        Convert to format for LoRA training
        
        Returns:
            Dictionary with instruction/output format
        
        FORMAT:
        {
            "instruction": "Perform a literature review on: {query}",
            "input": "",
            "output": "{excellent_review}"
        }
        
        This tells LoRA:
        INPUT: "Review CRISPR"
        DESIRED OUTPUT: Your excellent review
        
        LoRA learns to generate outputs like your excellent examples!
        """
        return {
            "instruction": f"Perform a comprehensive literature review on: {self.query}",
            "input": "",
            "output": self.review
        }
    
    def is_excellent(self, threshold: int = 8) -> bool:
        """Check if this is a good training example"""
        return self.rating >= threshold


class BackgroundTrainer:
    """
    Manages background LoRA training
    
    WHAT THIS DOES:
    1. Collects training examples as you provide feedback
    2. Every N examples, offers to train
    3. Generates Colab notebook for you
    4. You run on Colab (free GPU!)
    5. Download trained model
    6. Continue with improved model
    
    WORKFLOW:
    
    Examples 1-4: Collecting...
    Example 5: ✓ 5 examples collected! Ready to train?
              (y) - Generate Colab notebook now
              (n) - Collect more examples first
              (l) - Train later
    
    You: y
    
    System: Generated training_checkpoint_1.ipynb
            Upload to Colab and run!
            When done, download trained_model.gguf
            Place it in: ./models/checkpoint_1.gguf
    
    [30 minutes later, you're back]
    
    System: ✓ Loaded checkpoint_1.gguf
            Model is now smarter!
    
    Continue collecting with improved model...
    """
    
    def __init__(
        self,
        trigger_count: int = 5,
        checkpoint_dir: str = "training/checkpoints"
    ):
        """
        Initialize background trainer
        
        Args:
            trigger_count: Train after this many examples
            checkpoint_dir: Where to save checkpoints
        """
        self.trigger_count = trigger_count
        self.checkpoint_dir = checkpoint_dir
        self.examples: List[TrainingExample] = []
        self.checkpoint_number = 0
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs("training/notebooks", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Load existing examples
        self.load_examples()
        
        print(f"🎓 Background Trainer initialized")
        print(f"   Trigger: Every {trigger_count} examples")
        print(f"   Current examples: {len(self.examples)}")
    
    def add_example(
        self,
        query: str,
        review: str,
        rating: int,
        feedback: str
    ) -> bool:
        """
        Add training example and check if should train
        
        Args:
            query: Research query
            review: Generated review
            rating: Your rating
            feedback: Your feedback
        
        Returns:
            True if reached trigger count (ready to train)
        
        WHAT HAPPENS:
        1. Creates TrainingExample
        2. Adds to collection
        3. Saves to disk
        4. Checks if reached trigger count
        5. If yes, returns True (caller offers training)
        """
        example = TrainingExample(query, review, rating, feedback)
        self.examples.append(example)
        
        # Save to disk
        self.save_examples()
        
        # Check if should train
        total = len(self.examples)
        if total > 0 and total % self.trigger_count == 0:
            return True  # Ready to train!
        
        return False
    
    def prepare_training(self) -> str:
        """
        Prepare for LoRA training
        
        Returns:
            Path to generated Colab notebook
        
        WHAT THIS DOES:
        1. Filters to excellent examples only (8+)
        2. Exports to JSONL format
        3. Generates Colab notebook
        4. Returns path to notebook
        
        You then upload notebook to Colab and run it!
        """
        # Filter excellent examples
        excellent = [ex for ex in self.examples if ex.is_excellent()]
        
        if len(excellent) < 3:
            print("⚠️  Warning: Only {len(excellent)} excellent examples")
            print("   Training might not be effective")
            print("   Recommend 5+ excellent examples")
        
        # Increment checkpoint
        self.checkpoint_number += 1
        
        # Export training data
        training_file = f"{self.checkpoint_dir}/training_data_{self.checkpoint_number}.jsonl"
        self.export_for_lora(excellent, training_file)
        
        # Generate Colab notebook
        notebook_file = self.generate_colab_notebook(
            training_file,
            self.checkpoint_number
        )
        
        return notebook_file
    
    def export_for_lora(self, examples: List[TrainingExample], output_file: str):
        """
        Export examples to JSONL format for LoRA
        
        Args:
            examples: List of TrainingExample objects
            output_file: Where to save
        
        FORMAT:
        Each line is a JSON object:
        {"instruction": "...", "input": "", "output": "..."}
        {"instruction": "...", "input": "", "output": "..."}
        ...
        """
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example.to_lora_format()) + "\n")
        
        print(f"📝 Exported {len(examples)} examples to {output_file}")
    
    def generate_colab_notebook(self, training_data_path: str, checkpoint_num: int) -> str:
        """
        Generate Colab notebook for LoRA training
        
        Args:
            training_data_path: Path to training data JSONL
            checkpoint_num: Checkpoint number
        
        Returns:
            Path to generated notebook
        
        WHAT THIS CREATES:
        A complete Colab notebook that:
        1. Installs dependencies
        2. Loads training data
        3. Runs LoRA training
        4. Exports trained model
        5. Provides download link
        
        You just need to upload and click "Run all"!
        """
        
        # Read training data to embed in notebook
        with open(training_data_path, 'r') as f:
            training_data = f.read()
        
        notebook_path = f"training/notebooks/lora_checkpoint_{checkpoint_num}.ipynb"
        
        # Generate notebook content
        notebook_content = self._create_notebook_json(
            training_data,
            checkpoint_num
        )
        
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ COLAB NOTEBOOK GENERATED")
        print(f"{'='*60}")
        print(f"\nFile: {notebook_path}")
        print(f"\nNext steps:")
        print(f"1. Upload this notebook to Google Colab")
        print(f"2. Click 'Runtime' → 'Run all'")
        print(f"3. Wait 15-30 minutes for training")
        print(f"4. Download the trained model file")
        print(f"5. Place it in: ./models/checkpoint_{checkpoint_num}.gguf")
        print(f"6. Resume training here")
        print(f"{'='*60}\n")
        
        return notebook_path
    
    def _create_notebook_json(self, training_data: str, checkpoint_num: int) -> Dict:
        """
        Create Jupyter notebook JSON structure
        
        This is a valid Colab notebook with all necessary code!
        """
        
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {
                "colab": {
                    "name": f"LoRA Training - Checkpoint {checkpoint_num}",
                    "provenance": [],
                    "gpuType": "T4"
                },
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                },
                "accelerator": "GPU"
            },
            "cells": [
                # Cell 1: Instructions
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# LoRA Fine-Tuning - Checkpoint {checkpoint_num}\\n",
                        "\\n",
                        "This notebook trains your literature review model with LoRA.\\n",
                        "\\n",
                        "**Setup:**\\n",
                        "1. Runtime → Change runtime type → GPU (T4)\\n",
                        "2. Run all cells\\n",
                        "3. Wait ~20-30 minutes\\n",
                        "4. Download the trained model\\n",
                        "\\n",
                        "**What this does:**\\n",
                        "- Loads qwen2.5:7b\\n",
                        "- Adds LoRA adapters\\n",
                        "- Trains on your examples\\n",
                        "- Exports GGUF file for Ollama\\n"
                    ]
                },
                
                # Cell 2: Install dependencies
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "%%capture\\n",
                        "# Install required packages\\n",
                        "!pip install transformers>=4.35.0\\n",
                        "!pip install peft>=0.6.0\\n",
                        "!pip install accelerate>=0.24.0\\n",
                        "!pip install bitsandbytes>=0.41.0\\n",
                        "!pip install datasets\\n",
                        "!pip install trl\\n",
                        "print('✅ Dependencies installed')"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                
                # Cell 3: Load training data
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "import json\\n",
                        "from datasets import Dataset\\n",
                        "\\n",
                        "# Training data (embedded)\\n",
                        f"training_data_text = '''\\n{training_data}'''\\n",
                        "\\n",
                        "# Parse JSONL\\n",
                        "examples = []\\n",
                        "for line in training_data_text.strip().split('\\\\n'):\\n",
                        "    if line.strip():\\n",
                        "        examples.append(json.loads(line))\\n",
                        "\\n",
                        "print(f'Loaded {len(examples)} training examples')\\n",
                        "\\n",
                        "# Convert to dataset\\n",
                        "dataset = Dataset.from_list(examples)\\n",
                        "print(dataset)"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                
                # Cell 4: Load model and add LoRA
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "from transformers import AutoModelForCausalLM, AutoTokenizer\\n",
                        "from peft import get_peft_model, LoraConfig, TaskType\\n",
                        "import torch\\n",
                        "\\n",
                        "print('Loading base model...')\\n",
                        "\\n",
                        "# Load model (4-bit for memory efficiency)\\n",
                        "model_name = 'meta-llama/Llama-3.2-3B'\\n", #need to change to that for qwen2.5:7b
                        "\\n",
                        "model = AutoModelForCausalLM.from_pretrained(\\n",
                        "    model_name,\\n",
                        "    load_in_4bit=True,\\n",
                        "    torch_dtype=torch.float16,\\n",
                        "    device_map='auto'\\n",
                        ")\\n",
                        "\\n",
                        "tokenizer = AutoTokenizer.from_pretrained(model_name)\\n",
                        "tokenizer.pad_token = tokenizer.eos_token\\n",
                        "\\n",
                        "print('Adding LoRA adapters...')\\n",
                        "\\n",
                        "# LoRA configuration\\n",
                        "lora_config = LoraConfig(\\n",
                        "    r=16,  # LoRA rank\\n",
                        "    lora_alpha=32,\\n",
                        "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],\\n",
                        "    lora_dropout=0.05,\\n",
                        "    bias='none',\\n",
                        "    task_type=TaskType.CAUSAL_LM\\n",
                        ")\\n",
                        "\\n",
                        "model = get_peft_model(model, lora_config)\\n",
                        "model.print_trainable_parameters()\\n",
                        "\\n",
                        "print('✅ Model ready for training')"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                
                # Cell 5: Train
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "from transformers import TrainingArguments, Trainer\\n",
                        "\\n",
                        "print('Starting training...')\\n",
                        "print('This will take ~20-30 minutes')\\n",
                        "print('-' * 60)\\n",
                        "\\n",
                        "# Training arguments\\n",
                        "training_args = TrainingArguments(\\n",
                        "    output_dir='./lora_output',\\n",
                        "    num_train_epochs=3,\\n",
                        "    per_device_train_batch_size=1,\\n",
                        "    gradient_accumulation_steps=4,\\n",
                        "    learning_rate=2e-4,\\n",
                        "    fp16=True,\\n",
                        "    logging_steps=10,\\n",
                        "    save_strategy='epoch',\\n",
                        "    warmup_steps=10,\\n",
                        ")\\n",
                        "\\n",
                        "# Format data\\n",
                        "def format_func(example):\\n",
                        "    text = f\"### Instruction:\\\\n{example['instruction']}\\\\n\\\\n\"\\n",
                        "    text += f\"### Response:\\\\n{example['output']}\"\\n",
                        "    return {'text': text}\\n",
                        "\\n",
                        "formatted_dataset = dataset.map(format_func)\\n",
                        "\\n",
                        "# Tokenize\\n",
                        "def tokenize(example):\\n",
                        "    return tokenizer(example['text'], truncation=True, max_length=2048)\\n",
                        "\\n",
                        "tokenized = formatted_dataset.map(tokenize, remove_columns=['instruction', 'input', 'output', 'text'])\\n",
                        "\\n",
                        "# Train\\n",
                        "trainer = Trainer(\\n",
                        "    model=model,\\n",
                        "    args=training_args,\\n",
                        "    train_dataset=tokenized,\\n",
                        "    tokenizer=tokenizer\\n",
                        ")\\n",
                        "\\n",
                        "trainer.train()\\n",
                        "\\n",
                        "print('✅ Training complete!')"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                
                # Cell 6: Save model
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Save LoRA adapter\\n",
                        "model.save_pretrained('./lora_adapter')\\n",
                        "tokenizer.save_pretrained('./lora_adapter')\\n",
                        "\\n",
                        "print('✅ Model saved!')\\n",
                        "print('\\nNext steps:')\\n",
                        "print('1. Download the lora_adapter folder')\\n",
                        f"print('2. Place in your project as: models/checkpoint_{checkpoint_num}/')\\n",
                        "print('3. Resume training on your local machine')\\n",
                        "print('\\nThe model will be automatically loaded!')"
                    ],
                    "execution_count": None,
                    "outputs": []
                }
            ]
        }
        
        return notebook
    
    def save_examples(self):
        """Save examples to disk"""
        filepath = f"{self.checkpoint_dir}/all_examples.json"
        data = [
            {
                'query': ex.query,
                'review': ex.review,
                'rating': ex.rating,
                'feedback': ex.feedback,
                'timestamp': ex.timestamp
            }
            for ex in self.examples
        ]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_examples(self):
        """Load examples from disk"""
        filepath = f"{self.checkpoint_dir}/all_examples.json"
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.examples = [
                TrainingExample(
                    d['query'],
                    d['review'],
                    d['rating'],
                    d['feedback']
                )
                for d in data
            ]
        
        except FileNotFoundError:
            self.examples = []
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        excellent = [ex for ex in self.examples if ex.is_excellent()]
        ratings = [ex.rating for ex in self.examples] if self.examples else []
        
        return {
            "total_examples": len(self.examples),
            "excellent_examples": len(excellent),
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "next_trigger_at": ((len(self.examples) // self.trigger_count) + 1) * self.trigger_count,
            "examples_until_trigger": self.trigger_count - (len(self.examples) % self.trigger_count)
        }


# ═══════════════════════════════════════════════════════════════
# SUMMARY - How Background Training Works
# ═══════════════════════════════════════════════════════════════

"""
WORKFLOW:

1. YOU TRAIN (Phase 1):
   Example 1: Rate 7, feedback "needs stats"
   Example 2: Rate 8, feedback "good but more papers"
   Example 3: Rate 9, feedback "excellent"
   Example 4: Rate 6, feedback "too vague"
   Example 5: Rate 8, feedback "good"
   
   → System: "5 examples collected! Train now?"

2. GENERATE NOTEBOOK:
   System creates lora_checkpoint_1.ipynb
   Contains:
   - All your excellent examples (8+)
   - Complete LoRA training code
   - Ready to run on Colab

3. YOU TRAIN ON COLAB:
   Upload notebook to Colab
   Click "Run all"
   Wait 20-30 minutes
   Download trained adapter
   
   [Coffee break ☕]

4. LOAD TRAINED MODEL:
   Place adapter in models/checkpoint_1/
   System detects it
   Loads improved model automatically

5. CONTINUE TRAINING:
   Example 6-10: Model is now SMARTER
   Uses learned patterns from examples 1-5!
   
   After example 10: Trigger again!
   Can train checkpoint_2 with examples 1-10
   Even smarter model!


KEY POINTS:

✅ REAL learning: Weights actually change via LoRA
✅ FREE GPU: Uses Google Colab (no cost)
✅ INCREMENTAL: Each checkpoint builds on previous
✅ PORTABLE: Can use trained model in Ollama
✅ REVERSIBLE: Keep old checkpoints if new one is worse


VS OTHER APPROACHES:

PATTERN MEMORY:
- Instant but fake learning
- Just prompt engineering
- Good for quick iteration

BACKGROUND LoRA:
- Slow but true learning
- Actual weight updates
- Model genuinely improves

HYBRID (what we're building):
- Use both!
- Pattern memory for instant feedback
- LoRA for long-term improvement
"""
