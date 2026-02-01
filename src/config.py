import os
from pathlib import Path

from dotenv import load_dotenv

# locate & load
env_path = Path(__file__).parents[1] / '.env'
load_dotenv(env_path)


class Config:
    def __init__(self):
        self.debug = self._parse_bool(os.getenv('DEBUG'), default=False)
        self.insigma = self._parse_bool(os.getenv('INSIGMA'), default=False)
        self.flask_debug = self._parse_bool(os.getenv('FLASK_DEBUG'), default=False)
        self.tool_rag_port = self._parse_port(os.getenv('TOOL_RAG_PORT'), 8091)
        self.tool_haiku_port = self._parse_port(os.getenv('TOOL_HAIKU_PORT'), 8092)

        self.gigachat_base_url = os.getenv('GIGACHAT_BASE_URL')
        self.gigachat_cert_path = os.getenv('GIGACHAT_CERT_PATH')
        self.gigachat_key_path = os.getenv('GIGACHAT_KEY_PATH')
        self.gigachat_chain_path = os.getenv('GIGACHAT_CHAIN_PATH') or False

        self.openrouter_base_url = os.getenv('OPENROUTER_BASE_URL')
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        self.model = os.getenv('MODEL', 'openai/gpt-3.5-turbo')

        self.freezing = 1e-3

        self.validate()

    def _parse_bool(self, value: str | None, default: bool) -> bool:
        """
        Parse bool env values.
        """
        if value is None:
            return default
        return value.strip().lower() in {'1', 'true', 'yes'}

    def _parse_port(self, value: str | None, default: int) -> int:
        """
        Parse port env values.
        """
        if not value:
            return default
        try:
            return int(value)
        except ValueError as ex:
            raise ValueError(f'Invalid port value: {value}') from ex

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

        # check : type, sanity
        if not isinstance(self.tool_rag_port, int) or not isinstance(
            self.tool_haiku_port, int
        ):
            raise ValueError('Tool ports must be integers')
        if self.tool_rag_port == self.tool_haiku_port:
            raise ValueError('Tool ports must be different')

        # check : files exist
        if self.insigma:
            assert os.path.exists(self.gigachat_cert_path)
            assert os.path.exists(self.gigachat_key_path)

            if self.gigachat_chain_path:
                assert os.path.exists(self.gigachat_chain_path)


config = Config()
