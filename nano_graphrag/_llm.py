import numpy as np
from openai import AsyncOpenAI

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage


async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # Add retry logic with exponential backoff for rate limits
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                import asyncio
                import random
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit for {model}, waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                continue
            elif attempt < max_retries - 1:
                import asyncio
                print(f"API error for {model}: {e}, retrying...")
                await asyncio.sleep(1)
                continue
            else:
                raise e

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = AsyncOpenAI()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
