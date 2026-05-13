from providers.base import BaseSttAgent, BaseSttConfig as BaseSttConfig


def create_agent(provider: str) -> BaseSttAgent:
    if provider == "gladia":
        from providers.gladia import GladiaSttAgent, gladia_config

        return GladiaSttAgent(gladia_config)
    if provider == "openai":
        from providers.openai import OpenAiSttAgent, openai_config

        return OpenAiSttAgent(openai_config)
    if provider == "mistral":
        from providers.mistral import MistralSttAgent, mistral_config

        return MistralSttAgent(mistral_config)
    raise ValueError(f"Unknown STT provider: {provider}")
