from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

class LocalLLM:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.generator = None
        print(f"Initializing LocalLLM with model: {model_name}")

    def load_model(self):
        """Lazy load the model to avoid startup lag."""
        if self.generator is None:
            print(f"Loading {self.model_name}...")
            try:
                # Use 'text-generation' pipeline
                # device_map="auto" allows using GPU if available, else CPU
                self.generator = pipeline(
                    "text-generation", 
                    model=self.model_name,
                    device_map="auto" if torch.cuda.is_available() else "cpu"
                )
                print(f"{self.model_name} loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise e

    def generate(self, prompt, max_length=150):
        """Generate text from prompt."""
        self.load_model() # Ensure loaded
        
        try:
            # Generate
            response = self.generator(
                prompt, 
                max_length=max_length, 
                num_return_sequences=1,
                truncation=True,
                pad_token_id=50256 # GPT-2 pad token
            )
            
            # Extract text
            generated_text = response[0]['generated_text']
            
            # Simple cleanup: remove the prompt part if it's repeated (common in some pipelines)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
            return generated_text if generated_text else "..."
            
        except Exception as e:
            print(f"Generation error: {e}")
            return f"Error generating response: {str(e)}"

# Singleton instance
local_engine = LocalLLM()
