from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TextGenerator:
    def __init__(self):
        # Use GPT2-medium for better quality responses
        self.model_name = "gpt2-medium"  
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left',
                pad_token='<|endoftext|>'
            )
            
            self.device = "cpu"
            
            # Load model with optimized settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            ).to(self.device)
            
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
                
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        try:
            # Simplified and focused prompt template
            formatted_prompt = (
                f"Below is a topic. Provide a clear and factual response about it.\n\n"
                f"Topic: {prompt}\n\n"
                f"Response: "
            )
            
            # Tokenize with optimized parameters
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256  # Reduced for better focus
            ).to(self.device)
            
            # Generate with strict parameters
            outputs = self.model.generate(
                **inputs,
                max_length=min(max_length, 300),  # Further reduced max length
                num_return_sequences=1,
                temperature=0.5,    # Lower temperature for more deterministic output
                top_p=0.7,         # Reduced for more focused sampling
                do_sample=True,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.pad_token_id,
                min_length=30,      
                repetition_penalty=1.5,  # Increased to prevent loops
                length_penalty=1.0,      # Neutral length penalty
                early_stopping=True
            )
            
            # Clean up the response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "Response:" in generated_text:
                generated_text = generated_text.split("Response:")[-1].strip()
            
            return generated_text
            
        except Exception as e:
            raise Exception(f"Text generation failed: {str(e)}")

# Initialize the model
text_generator = TextGenerator()