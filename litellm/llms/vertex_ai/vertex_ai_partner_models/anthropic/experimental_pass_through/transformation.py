from typing import Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_logger
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_BETA_HEADER_VALUES,
    ANTHROPIC_HOSTED_TOOLS,
    ANTHROPIC_PROMPT_CACHING_SCOPE_BETA_HEADER,
)
from litellm.types.llms.anthropic_tool_search import get_tool_search_beta_header
from litellm.types.llms.vertex_ai import VertexPartnerProvider
from litellm.types.router import GenericLiteLLMParams

from ....vertex_llm_base import VertexBase


class VertexAIPartnerModelsAnthropicMessagesConfig(AnthropicMessagesConfig, VertexBase):
    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        """
        OPTIONAL

        Validate the environment for the request
        """
        if "Authorization" not in headers:
            vertex_ai_project = VertexBase.get_vertex_ai_project(litellm_params)
            vertex_credentials = VertexBase.get_vertex_ai_credentials(litellm_params)
            vertex_ai_location = VertexBase.get_vertex_ai_location(litellm_params)

            access_token, project_id = self._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_ai_project,
                custom_llm_provider="vertex_ai",
            )

            headers["Authorization"] = f"Bearer {access_token}"

            api_base = self.get_complete_vertex_url(
                custom_api_base=api_base,
                vertex_location=vertex_ai_location,
                vertex_project=vertex_ai_project,
                project_id=project_id,
                partner=VertexPartnerProvider.claude,
                stream=optional_params.get("stream", False),
                model=model,
            )

        headers["content-type"] = "application/json"
        
        # Add beta headers for Vertex AI
        tools = optional_params.get("tools", [])
        beta_values: set[str] = set()
        
        # Get existing beta headers if any
        existing_beta = headers.get("anthropic-beta")
        if existing_beta:
            beta_values.update(b.strip() for b in existing_beta.split(","))
        
        # Use the helper to remove unsupported beta headers
        self.remove_unsupported_beta(headers)
        beta_values.discard(ANTHROPIC_PROMPT_CACHING_SCOPE_BETA_HEADER)

        # Check for web search tool
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type", "").startswith(ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value):
                beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.WEB_SEARCH_2025_03_05.value)
                break
        
        # Check for tool search tools - Vertex AI uses different beta header
        anthropic_model_info = AnthropicModelInfo()
        if anthropic_model_info.is_tool_search_used(tools):
            beta_values.add(get_tool_search_beta_header("vertex_ai"))
        
        if beta_values:
            headers["anthropic-beta"] = ",".join(beta_values)
        
        return headers, api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            raise ValueError(
                "api_base is required. Unable to determine the correct api_base for the request."
            )
        return api_base  # no transformation is needed - handled in validate_environment

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        anthropic_messages_request = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        anthropic_messages_request["anthropic_version"] = "vertex-2023-10-16"

        anthropic_messages_request.pop(
            "model", None
        )  # do not pass model in request body to vertex ai

        anthropic_messages_request.pop(
            "output_format", None
        )  # do not pass output_format in request body to vertex ai - vertex ai does not support output_format as yet

        # Inject cache_control into the last message block for prompt caching support
        # on Vertex AI. Claude Code and other clients rely on the proxy to add this.
        # Only inject if no cache_control is already present in any message.
        # Related issue: https://github.com/BerriAI/litellm/issues/20418
        request_messages = anthropic_messages_request.get("messages")
        if request_messages and not self._has_cache_control_in_messages(
            request_messages
        ):
            self._inject_cache_control_to_last_message(request_messages)

        return anthropic_messages_request

    @staticmethod
    def _has_cache_control_in_messages(
        messages: List[Dict],
    ) -> bool:
        """
        Check if any message content block already has cache_control set.

        This prevents double-injection when the client has already added
        cache_control to their messages.
        """
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        return True
            elif isinstance(content, str):
                # String content can have cache_control at the message level
                if "cache_control" in message:
                    return True
        return False

    @staticmethod
    def _inject_cache_control_to_last_message(
        messages: List[Dict],
    ) -> None:
        """
        Inject cache_control into the last content block of the last message.

        For Vertex AI with Claude models, adding cache_control to the last
        message block enables prompt caching for all preceding content.
        This is required because Claude Code (v2.0.76+) no longer adds
        cache_control itself, relying on the proxy to handle it.

        Per Anthropic's specification, cache_control should only be placed
        on the last content block to create a single cache breakpoint.

        Args:
            messages: List of Anthropic-format message dicts. Modified in place.
        """
        if not messages:
            return

        last_message = messages[-1]
        content = last_message.get("content")

        if isinstance(content, list) and len(content) > 0:
            # Add cache_control to the last content block in the list
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}
                verbose_logger.debug(
                    "VertexAI Anthropic Messages: Injected cache_control to last "
                    "content block of last message for prompt caching"
                )
        elif isinstance(content, str):
            # For string content, convert to list format with cache_control
            last_message["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            verbose_logger.debug(
                "VertexAI Anthropic Messages: Converted string content to list "
                "and injected cache_control for prompt caching"
            )
    
    def remove_unsupported_beta(self, headers: dict) -> None:
        """
        Helper method to remove unsupported beta headers from the beta headers.
        Modifies headers in place.
        """
        unsupported_beta_headers = [
            ANTHROPIC_PROMPT_CACHING_SCOPE_BETA_HEADER
        ]
        existing_beta = headers.get("anthropic-beta")
        if existing_beta:
            filtered_beta = [
                b.strip()
                for b in existing_beta.split(",")
                if b.strip() not in unsupported_beta_headers
            ]
            if filtered_beta:
                headers["anthropic-beta"] = ",".join(filtered_beta)
            elif "anthropic-beta" in headers:
                del headers["anthropic-beta"]
