import yaml
from pathlib import Path

class Config:
    """Configuration manager for the pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set default paths
        self.data_dir = Path(self.config.get('paths', {}).get('data_dir', 'data'))
        self.output_dir = Path(self.config.get('paths', {}).get('output_dir', 'output'))
        self.model_dir = Path(self.config.get('paths', {}).get('model_dir', 'models'))
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Set model configurations
        self.models = self.config.get('models', {})
        
        # Processing parameters
        self.params = self.config.get('parameters', {})
        
    def get_param(self, section: str, param: str, default=None):
        """Get a specific parameter with fallback to default"""
        return self.config.get(section, {}).get(param, default)