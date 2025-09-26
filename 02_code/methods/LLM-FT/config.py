from argparse import ArgumentParser

class Config(object):
    def __init__(self):

        # General Params
        self.task = None
        self.output_dir = '/home/niklasdonhauser/LLM-Fine-Tune/results'
        self.data_path = '/home/niklasdonhauser/datasets'
        
        # Dataset Params
        self.dataset = None
        self.lang = 'en'
        self.eval_type = None
        self.data_setting = None
        
        # Training Params
        self.flash_attention = False
        self.quant = 4
        self.per_device_train_batch_size = 8
        self.gradient_accumulation_steps = 1
        self.learning_rate = None
        self.lr_scheduler_type = 'constant'
        self.num_train_epochs = None
        self.group_by_length = True
        self.logging_steps = 50
        self.save_strategy = 'epoch'
        self.evaluation_strategy = 'epoch'
        self.optim = 'paged_adamw_32bit'
        self.lora_r = None
        self.lora_alpha = None
        self.lora_dropout = 0.05
        self.from_checkpoint = False
        self.neftune_noise_alpha = None
        
        # Inference Params
        self.epoch = None
        self.per_device_eval_batch_size = 8
        self.max_new_tokens = 200
        self.top_k = -1
        self.top_p = 1
        self.temperature = 0
        self.gpu_memory_utilization = 0.8
        self.few_shots = None
        
        # Model Params
        self.model_name_or_path = "meta-llama/Meta-Llama-3-8B"
        self.seed = 42
        self.bf16 = False
        self.fp16 = True
        self.max_seq_length = 2048
        
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.update_config_with_args()

    def update_config_with_args(self):
        for key, value in self.args.items():
            if value is not None:
                setattr(self, key, value)
                
    def setup_parser(self):

        parser = ArgumentParser()

        # Model-related arguments
        parser.add_argument('--model_name_or_path', type=str, help='Base model name for training or inference')
        parser.add_argument('--seed', type=int, help='Seed to ensure reproducability.')
        parser.add_argument('--quant', type=int, help="How many bits to use for quantization.")
        # parser.add_argument('--bf16', action='store_true', help="Compute dtype of the model (uses bf16 if set).")
        parser.add_argument('--fp16', action='store_true', help="Use fp16 training.")
        parser.add_argument('--flash_attention', action='store_true', help='If to enable flash attention.')
        parser.add_argument('--epoch', type=int, help='Epoch checkpoint of the model.')
        parser.add_argument('--output_dir', type=str, help='Relative path to output directory.')
        # Dataset-related arguments
        parser.add_argument('--dataset', type=str, required=True, help="Which dataset to use: ['hotel', 'rest' or 'germeval']")
        parser.add_argument('--max_seq_length', type=int, help="Maximum context length during training and inference.")
        parser.add_argument('--task', type=str, default="acsa", help="Which ABSA Task the model was trained on. ['acd', 'acsa', 'acsd']")
        parser.add_argument('--eval_type', type=str, default="dev", help="Which dataset split used for evaluation")
        
         # Training arguments
        parser.add_argument('--per_device_train_batch_size', type=int, help='The training batch size per GPU.')
        parser.add_argument('--per_device_eval_batch_size', type=int, help='The evaluation batch size per GPU.')
        parser.add_argument('--gpu_memory_utilization', type=float, help='Percentage to which vllm can use GPU VRAM.')
        parser.add_argument('--gradient_accumulation_steps', type=int, help='Amount of gradients to accumulate before performing an optimizer step.')
        parser.add_argument('--learning_rate', type=float, help='The learning rate.')
        parser.add_argument('--lr_scheduler_type', type=str, help='Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis.')
        parser.add_argument('--group_by_length', action='store_true', help='Group sequences into batches with same length.')
        parser.add_argument('--num_train_epochs', type=int, help='Amount of epochs to train.')
        parser.add_argument('--logging_steps', type=int, help='The frequency of update steps after which to log the loss.')
        parser.add_argument('--save_strategy', type=str, help='When to save checkpoints.')
        parser.add_argument('--evaluation_strategy', type=str, help='When to compute eval loss on eval dataset.')
        parser.add_argument('--optim', type=str, help='The optimizer to be used.')
        parser.add_argument('--neftune_noise_alpha', type=int)

        # LORA-related arguments
        parser.add_argument('--lora_r', type=int, help="Lora R dimension.")
        parser.add_argument('--lora_alpha', type=int, help="Lora alpha.")
        parser.add_argument('--lora_dropout', type=float, help="Lora dropout.")
        parser.add_argument('--from_checkpoint', action='store_true', help="If resuming from checkpoint.")

        # Inference-related arguments
        parser.add_argument('--max_new_tokens', type=int, help="Maximum sequence length for new tokens during inference.")
        parser.add_argument('--top_k', type=float, help="Top-k sampling parameter.")
        parser.add_argument('--top_p', type=float, help="Top-p sampling parameter.")
        parser.add_argument('--temperature', type=float, help="Temperature for sampling.")
        parser.add_argument('--few_shots', type=int)
        
        return parser