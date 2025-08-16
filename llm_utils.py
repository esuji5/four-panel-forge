#!/usr/bin/env python3
"""
LLM Utils - 完全版（main.pyとの互換性を保持）

4コマ漫画解析のためのLLM統一インターフェース
既存のllm_utils.pyとの完全な後方互換性を提供
"""

import base64
import json
import os
import re
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anthropic
import google.genai as genai
import requests
from dotenv import load_dotenv
from google.genai.types import Part

# 環境変数読み込み
load_dotenv()


# タイムスタンプ付きログ関数
def log_with_time(message: str, level: str = "INFO"):
    """タイムスタンプ付きでログを出力"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    prefix = {
        "INFO": "ℹ️",
        "DEBUG": "🔍",
        "ERROR": "❌",
        "WARNING": "⚠️",
        "SUCCESS": "✅",
    }.get(level, "📝")

    log_message = f"[{timestamp}] {prefix} {message}"
    print(log_message)
    sys.stdout.flush()  # 即座に出力を反映


def load_config() -> Dict[str, Any]:
    """設定ファイルを読み込み（環境別）"""
    env = os.getenv("ENVIRONMENT", "local")
    config_dir = Path("config")

    # デフォルト設定を読み込み
    default_config = {}
    default_file = config_dir / "default.json"
    if default_file.exists():
        with open(default_file, "r", encoding="utf-8") as f:
            default_config = json.load(f)

    # 環境別設定を読み込み
    env_config = {}
    env_file = config_dir / f"{env}.json"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            env_config = json.load(f)

    # 設定をマージ（環境設定が優先）
    def merge_dict(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dict(result[key], value)
            else:
                result[key] = value
        return result

    return merge_dict(default_config, env_config)


# 正規表現パターン
RE_PAGE_KOMA_NUM = re.compile(r"\b\d{3}-\d\b")


@dataclass
class LLMConfig:
    """LLM設定クラス（環境別設定対応）"""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # デフォルトモデル（設定ファイルで上書き可能）
    openai_model: str = "gpt-4o-2024-11-20"
    anthropic_model: str = "claude-3-5-sonnet-20240620"
    gemini_model: str = "gemini-2.5-flash"

    # その他設定
    max_tokens: int = 4048
    confidence_threshold: float = 0.5
    temp_dir: str = "tmp/api_queries"

    def __post_init__(self):
        """設定ファイルから値を読み込み"""
        config = load_config()

        # LLM設定を読み込み
        llm_config = config.get("llm", {})

        # Gemini設定の上書き
        gemini_config = llm_config.get("gemini", {})
        if "model" in gemini_config:
            self.gemini_model = gemini_config["model"]

        # OpenAI設定の上書き
        openai_config = llm_config.get("openai", {})
        if "model" in openai_config:
            self.openai_model = openai_config["model"]
        if "max_tokens" in openai_config:
            self.max_tokens = openai_config["max_tokens"]

        # Anthropic設定の上書き
        anthropic_config = llm_config.get("anthropic", {})
        if "model" in anthropic_config:
            self.anthropic_model = anthropic_config["model"]

        # アプリ設定の上書き
        app_config = config.get("app", {})
        if "temp_dir" in app_config:
            self.temp_dir = app_config["temp_dir"]


@dataclass
class LLMResponse:
    """LLMレスポンス統一クラス"""

    content: str
    model: str
    provider: str
    timestamp: datetime
    metadata: Dict[str, Any]
    raw_response: Any


class ImageProcessor:
    """画像処理ユーティリティ"""

    @staticmethod
    def encode_image_to_base64(image_path: Union[str, Path]) -> str:
        """画像をBase64エンコード"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def clean_base64_data(base64_data: str) -> str:
        """Base64データのプレフィックスを除去"""
        if base64_data.startswith("data:image"):
            return base64_data.split(",")[1]
        return base64_data

    @staticmethod
    def save_base64_image(base64_data: str, output_path: Union[str, Path]) -> None:
        """Base64データを画像ファイルとして保存"""
        cleaned_data = ImageProcessor.clean_base64_data(base64_data)
        image_bytes = base64.b64decode(cleaned_data)
        with open(output_path, "wb") as f:
            f.write(image_bytes)


class QueryLogger:
    """APIクエリのログ機能"""

    def __init__(self, config: LLMConfig):
        self.config = config

    def create_query_dir(self, provider: str, suffix: str = "") -> Path:
        """クエリ保存用ディレクトリ作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        query_dir = Path(self.config.temp_dir) / f"{provider}_{suffix}_{timestamp}"
        query_dir.mkdir(parents=True, exist_ok=True)
        return query_dir

    def save_query_info(self, query_dir: Path, query_info: Dict[str, Any]) -> None:
        """クエリ情報を保存"""
        with open(query_dir / "query_info.json", "w", encoding="utf-8") as f:
            json.dump(query_info, f, ensure_ascii=False, indent=2)

    def save_prompt(self, query_dir: Path, prompt: str) -> None:
        """プロンプトを保存"""
        with open(query_dir / "prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)


class BaseLLMProvider(ABC):
    """LLMプロバイダーの基底クラス"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = QueryLogger(config)

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """レスポンス生成の抽象メソッド"""
        pass

    @abstractmethod
    def generate_image_response(self, prompt: str, image_path: str, **kwargs) -> LLMResponse:
        """画像付きレスポンス生成の抽象メソッド"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI APIプロバイダー"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.openai_api_key}",
        }

    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """テキストレスポンス生成"""
        model = kwargs.get("model", self.config.openai_model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return LLMResponse(
            content=content,
            model=model,
            provider="openai",
            timestamp=datetime.now(),
            metadata={"tokens_used": result.get("usage", {})},
            raw_response=result,
        )

    def generate_image_response(self, prompt: str, image_path: str, **kwargs) -> LLMResponse:
        """画像付きレスポンス生成"""
        model = kwargs.get("model", self.config.openai_model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # 画像をBase64エンコード
        base64_image = ImageProcessor.encode_image_to_base64(image_path)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }

        # クエリログ保存
        query_dir = self.logger.create_query_dir("openai", "image")
        self.logger.save_prompt(query_dir, prompt)
        shutil.copy2(image_path, query_dir / f"image{Path(image_path).suffix}")

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return LLMResponse(
            content=content,
            model=model,
            provider="openai",
            timestamp=datetime.now(),
            metadata={
                "tokens_used": result.get("usage", {}),
                "query_dir": str(query_dir),
            },
            raw_response=result,
        )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude APIプロバイダー"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """テキストレスポンス生成"""
        model = kwargs.get("model", self.config.anthropic_model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text

        return LLMResponse(
            content=content,
            model=model,
            provider="anthropic",
            timestamp=datetime.now(),
            metadata={
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            },
            raw_response=response,
        )

    def generate_image_response(self, prompt: str, image_path: str, **kwargs) -> LLMResponse:
        """画像付きレスポンス生成"""
        model = kwargs.get("model", self.config.anthropic_model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # 画像をBase64エンコード
        base64_image = ImageProcessor.encode_image_to_base64(image_path)

        # クエリログ保存
        query_dir = self.logger.create_query_dir("anthropic", "image")
        self.logger.save_prompt(query_dir, prompt)
        shutil.copy2(image_path, query_dir / f"image{Path(image_path).suffix}")

        message_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image,
                },
            },
        ]

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": message_content}],
        )

        content = response.content[0].text

        return LLMResponse(
            content=content,
            model=model,
            provider="anthropic",
            timestamp=datetime.now(),
            metadata={
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "query_dir": str(query_dir),
            },
            raw_response=response,
        )

    def generate_4koma_response(
        self, prompt: str, image_paths: List[str], image_data: Dict, **kwargs
    ) -> LLMResponse:
        """4コマ全体解析"""
        model = kwargs.get("model", self.config.anthropic_model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # メッセージコンテンツを構築
        message_content = []

        # 各画像を追加
        for i, image_path in enumerate(image_paths):
            message_content.append({"type": "text", "text": f"Image {i + 1}:"})
            base64_image = ImageProcessor.encode_image_to_base64("public" + image_path)
            message_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                }
            )

        # 画像データとプロンプトを追加
        message_content.append({"type": "text", "text": json.dumps(image_data)})
        message_content.append({"type": "text", "text": prompt})

        # クエリログ保存
        query_dir = self.logger.create_query_dir("anthropic", "4koma")
        self.logger.save_prompt(query_dir, prompt)

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": message_content}],
        )

        content = response.content[0].text

        return LLMResponse(
            content=content,
            model=model,
            provider="anthropic",
            timestamp=datetime.now(),
            metadata={
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "query_dir": str(query_dir),
            },
            raw_response=response,
        )


class GeminiProvider(BaseLLMProvider):
    """Google Gemini APIプロバイダー"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = genai.Client(api_key=config.google_api_key)

    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """テキストレスポンス生成"""
        model = kwargs.get("model", self.config.gemini_model)

        response = self.client.models.generate_content(model=model, contents=prompt)

        return LLMResponse(
            content=response.text,
            model=model,
            provider="gemini",
            timestamp=datetime.now(),
            metadata={"candidates": len(response.candidates)},
            raw_response=response,
        )

    def generate_image_response(self, prompt: str, image_path: str, **kwargs) -> LLMResponse:
        """画像付きレスポンス生成"""
        model = kwargs.get("model", self.config.gemini_model)

        log_with_time("🌐 Using Gemini API", level="INFO")
        # 画像をBase64エンコード
        base64_image = ImageProcessor.encode_image_to_base64(image_path)

        # クエリログ保存
        query_dir = self.logger.create_query_dir("gemini", "image")
        self.logger.save_prompt(query_dir, prompt)
        shutil.copy2(image_path, query_dir / f"image{Path(image_path).suffix}")

        # Partオブジェクトで画像付きコンテンツを作成
        contents = [
            Part(text=prompt),
            Part(inline_data={"mime_type": "image/jpeg", "data": base64_image}),
        ]

        response = self.client.models.generate_content(model=model, contents=contents)

        return LLMResponse(
            content=response.text,
            model=model,
            provider="gemini",
            timestamp=datetime.now(),
            metadata={
                "candidates": len(response.candidates),
                "query_dir": str(query_dir),
            },
            raw_response=response,
        )

    def generate_multimodal_response(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """マルチモーダルメッセージ処理"""
        model = kwargs.get("model", self.config.gemini_model)

        # クエリログ保存
        query_dir = self.logger.create_query_dir("gemini", "multimodal")

        # メッセージをPartオブジェクトに変換
        content_parts = []
        for i, message in enumerate(messages):
            if message.get("type") == "text":
                content_parts.append(Part(text=message["text"]))

                # テキストをファイルに保存
                with open(query_dir / f"message_{i}_text.txt", "w", encoding="utf-8") as f:
                    f.write(message["text"])

            elif message.get("type") == "image_url":
                image_url = message["image_url"]["url"]
                if image_url.startswith("data:image"):
                    base64_data = image_url.split(",")[1]
                    content_parts.append(
                        Part(inline_data={"mime_type": "image/jpeg", "data": base64_data})
                    )

                    # 画像をファイルに保存
                    ImageProcessor.save_base64_image(
                        base64_data, query_dir / f"message_{i}_image.jpg"
                    )

        response = self.client.models.generate_content(model=model, contents=content_parts)

        return LLMResponse(
            content=response.text,
            model=model,
            provider="gemini",
            timestamp=datetime.now(),
            metadata={
                "candidates": len(response.candidates),
                "query_dir": str(query_dir),
            },
            raw_response=response,
        )


class LLMManager:
    """LLMプロバイダー統一管理クラス"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.providers = {
            "openai": OpenAIProvider(self.config),
            "anthropic": AnthropicProvider(self.config),
            "gemini": GeminiProvider(self.config),
        }

    def get_provider(self, provider_name: str) -> BaseLLMProvider:
        """プロバイダー取得"""
        if provider_name not in self.providers:
            raise ValueError(f"Unsupported provider: {provider_name}")
        return self.providers[provider_name]

    def generate_response(self, provider: str, prompt: str, **kwargs) -> LLMResponse:
        """統一レスポンス生成インターフェース"""
        return self.get_provider(provider).generate_response(prompt, **kwargs)

    def generate_image_response(
        self, provider: str, prompt: str, image_path: str, **kwargs
    ) -> LLMResponse:
        """統一画像レスポンス生成インターフェース"""
        return self.get_provider(provider).generate_image_response(prompt, image_path, **kwargs)


# =============================================================================
# 以下、main.pyとの完全な後方互換性のためのレガシー関数
# =============================================================================

# グローバル設定インスタンス
_global_config = LLMConfig()


def encode_image(image_path):
    """レガシー: 画像をBase64エンコード"""
    return ImageProcessor.encode_image_to_base64(image_path)


def get_base64_encoded_image(image_path):
    """レガシー: 画像をBase64エンコード（別名）"""
    return ImageProcessor.encode_image_to_base64(image_path)


def fetch_response(prompt, image_path, model_name="gpt-4o-mini"):
    """レガシー: OpenAI APIで画像解析（main.pyで使用）"""
    log_with_time(f"start fetch_response {model_name}", level="INFO")

    # モデル名を修正（main.pyでハードコードされている）
    model_name = "gpt-4o-2024-11-20"

    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_global_config.openai_api_key}",
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 4000,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response


def fetch_gemini_response(prompt, image_path, model="gemini-2.5-flash"):
    """レガシー: Gemini画像処理（main.pyで使用）"""
    log_with_time(f"start fetch_gemini_response with {model}", level="INFO")

    log_with_time("🌐 Using Gemini API", level="INFO")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    query_dir = f"tmp/api_queries/gemini_{timestamp}"
    os.makedirs(query_dir, exist_ok=True)

    # 画像をbase64エンコード
    base64_image = encode_image(image_path)

    # Geminiクライアントを作成
    client = genai.Client(api_key=_global_config.google_api_key)

    # プロンプトを保存
    with open(f"{query_dir}/prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    # 元画像をコピー
    image_filename = os.path.basename(image_path)
    shutil.copy2(image_path, f"{query_dir}/{image_filename}")

    # クエリ情報を保存
    query_info = {
        "model": model,
        "timestamp": timestamp,
        "prompt_length": len(prompt),
        "image_path": image_path,
        "image_filename": image_filename,
        "base64_length": len(base64_image),
    }
    with open(f"{query_dir}/query_info.json", "w", encoding="utf-8") as f:
        json.dump(query_info, f, ensure_ascii=False, indent=2)

    log_with_time(f"Query saved to: {query_dir}", level="INFO")

    # 画像付きプロンプトを作成（Partオブジェクトを使用）
    contents = [
        Part(text=prompt),
        Part(inline_data={"mime_type": "image/jpeg", "data": base64_image}),
    ]

    # レスポンスを取得
    response = client.models.generate_content(model=model, contents=contents)

    return response


def fetch_gemini_response_base64(
    prompt, base64_image, model="gemini-2.5-flash", original_image_path=None
):
    """レガシー: Gemini Base64画像処理（main.pyで使用）"""
    log_with_time(f"start fetch_gemini_response_base64 with {model}", level="INFO")

    log_with_time("🌐 Using Gemini API", level="INFO")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    query_dir = f"tmp/api_queries/gemini_base64_{timestamp}"
    os.makedirs(query_dir, exist_ok=True)

    # Geminiクライアントを作成
    client = genai.Client(api_key=_global_config.google_api_key)

    # base64_imageがdata:image/jpeg;base64,で始まる場合は削除
    if base64_image.startswith("data:image"):
        base64_image = base64_image.split(",")[1]

    # プロンプトを保存
    with open(f"{query_dir}/prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    # 画像を保存（Base64デコードして保存）
    import base64 as b64

    image_data = b64.b64decode(base64_image)
    with open(f"{query_dir}/image.jpg", "wb") as f:
        f.write(image_data)

    # クエリ情報を保存
    query_info = {
        "model": model,
        "timestamp": timestamp,
        "prompt_length": len(prompt),
        "base64_length": len(base64_image),
        "image_size": len(image_data),
        "original_image_path": original_image_path,
    }
    with open(f"{query_dir}/query_info.json", "w", encoding="utf-8") as f:
        json.dump(query_info, f, ensure_ascii=False, indent=2)

    log_with_time(f"Query saved to: {query_dir}", level="INFO")

    # 画像付きプロンプトを作成（Partオブジェクトを使用）
    contents = [
        Part(text=prompt),
        Part(inline_data={"mime_type": "image/jpeg", "data": base64_image}),
    ]

    # レスポンスを取得
    response = client.models.generate_content(model=model, contents=contents)

    return response


def fetch_gemini_response_multimodal(prompt_messages, model="gemini-2.5-flash"):
    """レガシー: Gemini APIで配列形式のプロンプトメッセージを処理（main.pyで使用）"""
    log_with_time(f"start fetch_gemini_response_multimodal with {model}", level="INFO")
    log_with_time(f"Processing {len(prompt_messages)} messages", level="INFO")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    query_dir = f"tmp/api_queries/gemini_multimodal_{timestamp}"
    os.makedirs(query_dir, exist_ok=True)

    # クエリ情報を保存
    query_info = {
        "model": model,
        "timestamp": timestamp,
        "message_count": len(prompt_messages),
        "messages": [],
    }

    # Geminiクライアントを作成
    client = genai.Client(api_key=_global_config.google_api_key)

    # プロンプトメッセージをGeminiの形式に変換
    content_parts = []

    for i, message in enumerate(prompt_messages):
        if message.get("type") == "text":
            # テキストのPartオブジェクトを作成
            text_content = message["text"]
            content_parts.append(Part(text=text_content))
            log_with_time(f"Message {i}: text ({len(text_content)} chars)", level="INFO")

            # テキストをファイルに保存
            text_file = f"{query_dir}/message_{i}_text.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text_content)

            query_info["messages"].append(
                {
                    "index": i,
                    "type": "text",
                    "file": f"message_{i}_text.txt",
                    "length": len(text_content),
                }
            )

        elif message.get("type") == "image_url":
            # image_urlから画像データを抽出
            image_url = message["image_url"]["url"]
            if image_url.startswith("data:image"):
                # "data:image/jpeg;base64," の部分を除去
                base64_data = image_url.split(",")[1]
                # 画像のPartオブジェクトを作成
                content_parts.append(
                    Part(inline_data={"mime_type": "image/jpeg", "data": base64_data})
                )
                log_with_time(
                    f"Message {i}: image (base64, {len(base64_data)} chars)",
                    level="INFO",
                )

                # 画像をファイルに保存（Base64デコードして保存）
                import base64 as b64

                image_data = b64.b64decode(base64_data)
                image_file = f"{query_dir}/message_{i}_image.jpg"
                with open(image_file, "wb") as f:
                    f.write(image_data)

                query_info["messages"].append(
                    {
                        "index": i,
                        "type": "image",
                        "file": f"message_{i}_image.jpg",
                        "base64_length": len(base64_data),
                        "file_size": len(image_data),
                    }
                )

            else:
                log_with_time(
                    f"Warning: Unsupported image URL format: {image_url[:50]}...",
                    level="WARNING",
                )
        else:
            log_with_time(
                f"Warning: Unsupported message type: {message.get('type')}",
                level="WARNING",
            )

    # クエリ情報をJSONで保存
    with open(f"{query_dir}/query_info.json", "w", encoding="utf-8") as f:
        json.dump(query_info, f, ensure_ascii=False, indent=2)

    log_with_time(f"Generated {len(content_parts)} content parts for Gemini", level="INFO")
    log_with_time(f"Query saved to: {query_dir}", level="INFO")

    # レスポンスを取得
    try:
        response = client.models.generate_content(model=model, contents=content_parts)
        return response
    except Exception as e:
        log_with_time(f"Error in Gemini API call: {e}", level="ERROR")
        log_with_time(f"Content parts: {len(content_parts)} items", level="ERROR")
        for i, part in enumerate(content_parts):
            log_with_time(f"  Part {i}: {type(part)}", level="ERROR")
            if hasattr(part, "text"):
                log_with_time(f"    - text: {len(part.text)} chars", level="ERROR")
            if hasattr(part, "inline_data"):
                log_with_time(
                    f"    - inline_data: {part.inline_data.keys() if isinstance(part.inline_data, dict) else 'present'}",
                    level="ERROR",
                )
        raise


def fetch_gemini_cli_response(prompt, model="gemini-2.5-flash"):
    """レガシー: Gemini APIでCLIからのプロンプトを処理（main.pyで使用）"""
    log_with_time(f"start fetch_gemini_cli_response with {model}", level="INFO")
    client = genai.Client(api_key=_global_config.google_api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response


def fetch_gemini_4koma_response(prompt, image_path_list, image_data, model="gemini-2.5-flash"):
    """レガシー: Gemini APIで4コマ全体を処理（main.pyで使用）"""
    log_with_time(f"start fetch_gemini_4koma_response with {model}", level="INFO")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    query_dir = f"tmp/api_queries/gemini_4koma_{timestamp}"
    os.makedirs(query_dir, exist_ok=True)

    # Geminiクライアントを作成
    client = genai.Client(api_key=_global_config.google_api_key)

    # 複数画像とプロンプトを作成（Partオブジェクトを使用）
    contents = []

    # クエリ情報を保存
    query_info = {
        "model": model,
        "timestamp": timestamp,
        "prompt": prompt,
        "image_count": len(image_path_list),
        "images": [],
    }

    # 各画像を追加
    for i, image_path in enumerate(image_path_list):
        contents.append(Part(text=f"Image {i + 1}:"))
        base64_image = get_base64_encoded_image("public" + image_path)
        contents.append(Part(inline_data={"mime_type": "image/jpeg", "data": base64_image}))

        # 画像を保存（Base64デコードして保存）
        import base64 as b64

        image_data_bytes = b64.b64decode(base64_image)
        image_file = f"{query_dir}/image_{i + 1}.jpg"
        with open(image_file, "wb") as f:
            f.write(image_data_bytes)

        query_info["images"].append(
            {
                "index": i + 1,
                "path": image_path,
                "file": f"image_{i + 1}.jpg",
                "size": len(image_data_bytes),
            }
        )

    # imageDataとプロンプトを追加
    contents.append(Part(text=json.dumps(image_data)))
    contents.append(Part(text=prompt))

    # プロンプトを保存
    with open(f"{query_dir}/prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    # imageDataを保存
    with open(f"{query_dir}/image_data.json", "w", encoding="utf-8") as f:
        json.dump(image_data, f, ensure_ascii=False, indent=2)

    # クエリ情報を保存
    with open(f"{query_dir}/query_info.json", "w", encoding="utf-8") as f:
        json.dump(query_info, f, ensure_ascii=False, indent=2)

    log_with_time(f"Query saved to: {query_dir}", level="INFO")

    # レスポンスを取得
    response = client.models.generate_content(model=model, contents=contents)

    return response


def fetch_gemini_discussion_response(prompt, summary, image_data, model="gemini-2.5-flash"):
    """レガシー: Gemini APIでディスカッション処理（main.pyで使用）"""
    log_with_time(f"start fetch_gemini_discussion_response with {model}", level="INFO")

    # Geminiクライアントを作成
    client = genai.Client(api_key=_global_config.google_api_key)

    # コンテンツを構築
    contents = []

    # summaryがテキストの場合とオブジェクトの場合の両方に対応
    if isinstance(summary, str):
        summary_text = summary
    elif isinstance(summary, dict) and "content" in summary:
        # Claudeのレスポンス形式の場合
        summary_text = (
            summary["content"][0]["text"]
            if isinstance(summary["content"], list)
            else summary["content"]
        )
    else:
        summary_text = json.dumps(summary)

    # 会話の流れを構築
    contents.append(Part(text="以下の4コマ漫画について話し合いましょう。"))
    contents.append(Part(text=f"画像データ: {json.dumps(image_data)}"))
    contents.append(Part(text=f"4コマ全体のまとめ: {summary_text}"))
    contents.append(Part(text=f"質問: {prompt}"))

    # レスポンスを取得
    response = client.models.generate_content(model=model, contents=contents)

    return response


def fetch_anthropic_4koma_response(prompt, image_path_list, image_data):
    """レガシー: Claude APIで4コマ全体を処理（main.pyで使用）"""
    MODEL_NAME = "claude-3-5-sonnet-20240620"
    log_with_time(f"start fetch_anthropic_4koma_response {MODEL_NAME}", level="INFO")
    message_list = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": get_base64_encoded_image("public" + image_path_list[0]),
                    },
                },
                {"type": "text", "text": "Image 2:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": get_base64_encoded_image("public" + image_path_list[1]),
                    },
                },
                {"type": "text", "text": "Image 3:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": get_base64_encoded_image("public" + image_path_list[2]),
                    },
                },
                {"type": "text", "text": "Image 4:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": get_base64_encoded_image("public" + image_path_list[3]),
                    },
                },
                {"type": "text", "text": json.dumps(image_data)},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    client = anthropic.Anthropic(
        api_key=_global_config.anthropic_api_key,
    )
    response = client.messages.create(model=MODEL_NAME, max_tokens=4048, messages=message_list)
    return response


def fetch_anthropic_discussion_response(prompt, summary, image_data):
    """レガシー: Claude APIでディスカッション処理（main.pyで使用）"""
    MODEL_NAME = "claude-3-5-sonnet-20240620"
    log_with_time(f"start fetch_anthropic_discussion_response {MODEL_NAME}", level="INFO")
    assistant_data = {}
    assistant_data = {k: v for k, v in summary.items() if k in ["role", "content"]}

    message_list = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(image_data)},
                {
                    "type": "text",
                    "text": "画像郡とimageDataから4コマ全体でどんな話かをまとめてください",
                },
            ],
        },
        assistant_data,
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    client = anthropic.Anthropic(
        api_key=_global_config.anthropic_api_key,
    )
    response = client.messages.create(model=MODEL_NAME, max_tokens=4048, messages=message_list)
    return response


def get_gpt_response(prompt: str, image_path: str, model_name: str = "gpt-4o-2024-11-20") -> Dict:
    """レガシー: GPTレスポンス取得（後方互換性）"""
    manager = LLMManager()
    response = manager.generate_image_response("openai", prompt, image_path, model=model_name)
    return {"choices": [{"message": {"content": response.content}}]}


def get_claude_response(prompt: str, image_path: str = None) -> Dict:
    """レガシー: Claudeレスポンス取得（後方互換性）"""
    manager = LLMManager()
    if image_path:
        response = manager.generate_image_response("anthropic", prompt, image_path)
    else:
        response = manager.generate_response("anthropic", prompt)
    return {"content": [{"text": response.content}]}


def get_gemini_response(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """レガシー: Geminiレスポンス取得（後方互換性）"""
    manager = LLMManager()
    response = manager.generate_response("gemini", prompt, model=model)
    return response.content


def print_and_save_response(response, koma_id):
    """レガシー: レスポンスの表示と保存"""
    save_dir = "chat_results"
    os.makedirs(name=save_dir, exist_ok=True)
    log_with_time(f"Response attributes: {dir(response)}", level="DEBUG")
    log_with_time(f"Response content: {response.json()['content'][0]['text']}", level="DEBUG")

    with open(save_dir + "/" + koma_id + ".json", "w") as f:
        f.write(response.text)


def save_response(response, image_path):
    """レガシー: レスポンスをファイルに追記"""
    p = Path(image_path)
    with open(f"chat_results/result.txt", "+a") as f:
        f.write(f"{image_path}___{response.json()}\n")


def build_koma_id(image_path):
    """レガシー: 画像パスからコマIDを生成"""
    data_name = image_path.split("/")[1]
    kanji = data_name[-2:]
    page_koma_num = RE_PAGE_KOMA_NUM.findall(image_path)[-1]
    koma_id = f"{kanji}-{page_koma_num}"
    return koma_id


# プロンプトテンプレート
prompt_anal_koma = """
「わからないことは知らないと答えて」
以下は『ゆゆ式』の画像です。日本の4コマ漫画のため、右から左に読んでいきます。主な登場人物は以下です。

- 日向縁
	- "縁の髪の特徴": [  "黒っぽい髪色",  "長め（肩くらいの長さ）",  "ストレートな髪質",  "前髪がある"  ]
- 櫟井唯 
	- "唯の髪の特徴": [  "白っぽい髪色",  "短め（首あたりの長さ）",  "ややウェーブがかかっているように見える",  "前髪がある",  "髪の先端が少し外側に跳ねている"  ]
- 野々原ゆずこ 
	- "ゆずこの髪の特徴": [ "短め",  "ボブカットのようなスタイル",  "前髪がやや長め",  "髪の先端が内側に少し巻き込んでいる",  "髪色は白髪でも黒髪でもない中間的な色（おそらくグレーがかった色）"  ]
この画像について以下に答えてください。それぞれについて画像の右側にあるものから順に答えてください。
- ゆずこたちは、それぞれがコマのどこにいますか？誰もいない場合もあるし、3人より多い人数が写っている場合もあります
	- 画像の右側にあるものから順に答えてください。
	- 座標も答えてください
- どんな表情をしていますか?いない場合は「いない」
- 顔の向きを答えてください.いない場合は「いない」
- どんなセリフを言っていますか？ない場合は「なし」
	- 画像の右側にあるものから順に答えてください。
	- 座標も画像に対する比率で答えてください で答えてください 
	- 文が複数ある場合は半角スペースで区切ってください
- どんな服装をしていますか？いない場合は「いない」
- どんな場面だと思われますか？いない場合は「いない」
- 話している場所はどこですか？わからない場合は「不明」
- 背景に漫画ならではの効果描写があれば教えて下さい。ない場合は「なし」
- 画像の中には全部で何人いますか
- 画像の中には吹き出しがいくつありますか
- 出力はJSON形式のみ

出力するjsonは以下の形式にしてください
```
{
    charactersNum: int,
    serifsNum: int,
    characters: [
    {
        character: "",
        faceDirection: "",
        position: "",
        expression: "",
        serif: "",
        clothing: "",
        isVisible: true,
    },
    {
        character: "",
        faceDirection: "",
        position: "",
        expression: "",
        serif: "",
        clothing: "",
        isVisible: true,
    },
    {
        character: "",
        faceDirection: "",
        position: "",
        expression: "",
        serif: "",
        clothing: "",
        isVisible: true,
    },
    ],
    sceneData: {scene: "", location: "", backgroundEffects: ""},
}
```
"""


# プロンプトテンプレート（クラス版）
class PromptTemplates:
    """4コマ漫画解析用プロンプトテンプレート"""

    KOMA_ANALYSIS = prompt_anal_koma


def save_response_to_file(response: LLMResponse, output_path: str) -> None:
    """レスポンスをファイルに保存"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "content": response.content,
                "model": response.model,
                "provider": response.provider,
                "timestamp": response.timestamp.isoformat(),
                "metadata": response.metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
