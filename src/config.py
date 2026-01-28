import os
from pathlib import Path

from dotenv import load_dotenv

# locate & load
env_path = Path(__file__).parents[1] / '.env'
load_dotenv(env_path)


class Config:
    def __init__(self):
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.insigma = os.getenv('INSIGMA', 'false').lower() == 'true'

        self.gigachat_base_url = os.getenv('GIGACHAT_BASE_URL')
        self.gigachat_cert_path = os.getenv('GIGACHAT_CERT_PATH')
        self.gigachat_key_path = os.getenv('GIGACHAT_KEY_PATH')
        self.gigachat_chain_path = os.getenv('GIGACHAT_CHAIN_PATH') or False

        self.openrouter_base_url = os.getenv('OPENROUTER_BASE_URL')
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        self.model = os.getenv('MODEL', 'openai/gpt-3.5-turbo')

        self.validate()

    def validate(self):
        required_vars = (
            [
                'GIGACHAT_BASE_URL',
                'GIGACHAT_CERT_PATH',
                'GIGACHAT_KEY_PATH',
            ]
            if self.insigma
            else ['OPENROUTER_API_KEY', 'OPENROUTER_BASE_URL']
        )
        # check : nonempty
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise EnvironmentError(
                'Missing required env vars: {}'.format(','.join(missing))
            )

        # check : files exist
        if self.insigma:
            assert os.path.exists(self.gigachat_cert_path)
            assert os.path.exists(self.gigachat_key_path)

            if self.gigachat_chain_path:
                assert os.path.exists(self.gigachat_chain_path)


config = Config()
