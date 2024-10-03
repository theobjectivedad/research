"""
Copyright 2024 theobjectivedad@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Dependencies:
# numpy Faker tiktoken

import json
import logging
import math
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Literal, Tuple, Union
from uuid import uuid4

import tiktoken
from faker import Faker
from pydantic import BaseModel, Field

LOG = logging.getLogger(__name__)


StatSortType = Literal["name", "calls", "tokens"]
StatusType = Literal["SUCCESS", "ERROR"]


ModelNameType = Literal["gpt4o", "gpt4o-mini", "o1-preview", "o1-mini", "sonnet-3.5"]

# Units are cost per 1k request/response tokens by model
# Reference: https://docsbot.ai/tools/gpt-openai-api-pricing-calculator
#
# Note that the current implementaton uses tiktoken to estimate token counts
_MODEL_PRICE_TABLE: Dict[ModelNameType, Tuple[Decimal, Decimal]] = {
    "gpt4o": (Decimal("0.0025"), Decimal("0.01")),
    "gpt4o-mini": (Decimal("0.00015"), Decimal("0.0006")),
    "o1-preview": (Decimal("0.015"), Decimal("0.06")),
    "o1-mini": (Decimal("0.003"), Decimal("0.012")),
    "sonnet-3.5": (Decimal("0.003"), Decimal("0.015")),
}


class SimCallLog(BaseModel):
    """
    A record of a single LLM generation, including the prompt, response, and metadata.
    """

    status: StatusType = Field(...)
    prompt: str = Field(...)
    request_tokens: int = Field(..., ge=0)
    response_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(..., ge=0)
    text: str | None = Field(...)
    n: int = Field(..., ge=0)
    prefix: str = Field(...)
    call_kwargs: Dict[str, Any] = Field(...)


class StatsRecord(BaseModel):
    """
    A record of LLM generation statistics for a given prefix.
    """

    prefix: str = Field(..., description="The prefix used in the LLM generation")
    total_count: int = Field(
        ..., ge=0, description="Number of times the prefix was used"
    )
    success_count: int = Field(
        ..., ge=0, description="Number of successful generations for the prefix"
    )
    error_count: int = Field(
        ..., ge=0, description="Number of failed generations for the prefix"
    )
    request_tokens: int = Field(
        ..., ge=0, description="Total request tokens in all requests for the prefix"
    )
    response_tokens: int = Field(
        ..., ge=0, description="Total response tokens in all responses for the prefix"
    )
    total_tokens: int = Field(..., ge=0, description="Total tokens for the prefix")


class ModelCostEstimation(BaseModel):
    """
    A record of the estimated costs for a given model.
    """

    llm_name: str = Field(..., description="The name of the model")

    total_cost: float = Field(..., ge=0, description="Total estimated cost")

    request_cost: float = Field(
        ..., ge=0, description="Estimated cost for request tokens"
    )
    response_cost: float = Field(
        ..., ge=0, description="Estimated cost for response tokens"
    )

    request_tokens: int = Field(
        ..., ge=0, description="Total request tokens in all requests for the model"
    )
    response_tokens: int = Field(
        ..., ge=0, description="Total response tokens in all responses for the model"
    )
    total_tokens: int = Field(..., ge=0, description="Total tokens for the model")

    generation_count: int = Field(
        ..., ge=0, description="Total number of LLM generations"
    )


class LLMSimulatorError(Exception):
    """Simulated generation call exception"""


class LLMSimulatorBase(ABC):
    """
    Simulates calling a Language Model (LLM) by generating a list of simulated
    responses from one or more rendered prompts.

    This class is a very loosely based on langchain_core.langchain_models.llms.BaseLLM
    """

    @abstractmethod
    def __call__(
        self,
        prompts: Union[List[str], str],
        *,
        n: int = 1,
        prefix: str = "GEN",
        append: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> List[str | None]:
        """
        Abstract method to generate simulated LLM responses.

        Args:
            prompts (Union[List[str], str]): Fully rendered prompt(s) that would be
                passed to the LLM.
            n (int, optional): Number of responses to generate for each prompt. Defaults
                to 1.
            prefix (str, optional): Prefix to add to each generated response. Defaults
                to "GEN".
            append (str, optional): String to append at the end of each generated
                response. Defaults to None.
            min_tokens (int, optional): Minimum number of tokens in each simulated LLM
                response. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens in each simulated LLM
                response. Defaults to None.
            **kwargs: Additional arguments, intended use is to document / capture
                additional model parameters passed to inferencing function

        Returns:
            List[str | None]: A list of generated responses, each corresponding to a
                prompt. When a response contains None, it represents a LLM inferencing
                error.
        """

    def invoke(
        self,
        prompt: str,
        *,
        n: int = 1,
        prefix: str = "GEN",
        append: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> List[str | None]:
        return self(
            prompt,
            n=n,
            prefix=prefix,
            append=append,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def ainvoke(
        self,
        prompt: str,
        *,
        n: int = 1,
        prefix: str = "GEN",
        append: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> List[str | None]:
        return self(
            prompt,
            n=n,
            prefix=prefix,
            append=append,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            **kwargs,
        )

    def batch(
        self,
        prompts: List[str],
        *,
        n: int = 1,
        prefix: str = "GEN",
        append: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> List[str | None]:
        return self(
            prompts,
            n=n,
            prefix=prefix,
            append=append,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def abatch(
        self,
        prompts: List[str],
        *,
        n: int = 1,
        prefix: str = "GEN",
        append: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> List[str | None]:
        return self(
            prompts,
            n=n,
            prefix=prefix,
            append=append,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            **kwargs,
        )

    @staticmethod
    def render_template(
        *, template: str, template_values: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> List[str]:
        """
        Substitutes placeholders in a template string with values from a dictionary or
        a list of dictionaries.

        Args:
            template (str): The template string containing placeholders in the format
                '{key}'.
            template_values (Union[List[Dict[str, Any]], Dict[str, Any]]): A dictionary
                or list of dictionaries where keys match the placeholders in the
                template, and values are the substitutions.

        Returns:
            List[str]: A list of rendered template strings with placeholders substituted
            by their corresponding values.

        Raises:
            ValueError: If a substitution pattern in 'template' is not found in
                'template_values', or if a key in 'template_values' is not found in
                'template'.
        """

        # Validate and normalize the template_values input
        if isinstance(template_values, Dict):
            template_values = [template_values]

        # Extract all substitution patterns from the template (placeholders in the form '{key}')
        template_keys = set(re.findall(r"{(.*?)}", template))
        rendered_templates = []

        for v_index, cur_template_value in enumerate(template_values):
            # Check for missing template keys in the current value dictionary
            missing_keys = template_keys - set(cur_template_value.keys())
            if missing_keys:
                raise ValueError(
                    f"Missing keys {missing_keys} in template_values at index {v_index}"
                )

            # Check for extra keys in the current value dictionary that are not used in the template
            extra_keys = set(cur_template_value.keys()) - template_keys
            if extra_keys:
                raise ValueError(
                    f"Extra keys {extra_keys} in template_values at index {v_index} not found in template"
                )

            # Render the template by replacing placeholders with their corresponding values
            rendered_template = template
            for key, value in cur_template_value.items():
                placeholder = f"{{{key}}}"
                rendered_template = rendered_template.replace(placeholder, str(value))

            rendered_templates.append(rendered_template.strip())

        return rendered_templates


class LLMSimulatorLite(LLMSimulatorBase):
    """
    Simple LLM simulator with minimal external dependencies that generates a fixed
    response for each prompt
    """

    def __init__(self, unique: bool = True, chars_per_token: int = 3):

        # When true, add a unique element to each prompt. This is useful when testing
        # applications that use set-style collections as part of their algorithm.
        self.unique = unique

        # When constructing responses of a desired length, this is the value that is
        # used to estimate the number of tokens per character.
        self.chars_per_token = chars_per_token

    def _estimate_tokens(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token)

    def __call__(
        self,
        prompts: Union[List[str], str],
        *,
        n: int = 1,
        prefix: str = "GEN",
        append: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> List[str | None]:

        if isinstance(prompts, str):
            prompts = [prompts]

        # Generate a fixed response for each prompt, assume 3 characters per token
        if min_tokens is not None and max_tokens is not None:
            target_tokens = random.randint(min_tokens, max_tokens)
        else:
            target_tokens = 100

        derived_append = append if append is not None else ""

        prefix_tokens = self._estimate_tokens(prefix)
        derived_append_tokens = self._estimate_tokens(derived_append)

        # Generate response strings for each prompt response value
        responses: List[str | None] = []
        for _ in range(n * len(prompts)):

            # Generate a unique UUID for each response if required
            if self.unique:
                derived_uuid = str(uuid4()).replace("-", "")
            else:
                derived_uuid = ""

            # Estimate the number of tokens for each part of the response
            uuid_tokens = self._estimate_tokens(derived_uuid)

            # Calculate how many filler tokens we need to generate
            filler_tokens = target_tokens - (
                uuid_tokens + prefix_tokens + derived_append_tokens
            )

            # Calculate how many filler characters we need to generate, note the -3 is
            # estimating for the whitespace in between response template fields below.
            filler_chars = math.ceil(filler_tokens * self.chars_per_token) - 3

            filler = "A" * filler_chars

            cur_response = f"{prefix},{derived_uuid},{filler},{derived_append}".strip()
            responses.append(cur_response)

        return responses


# pylint: disable=too-many-instance-attributes
class LLMSimulatorFull(LLMSimulatorBase):
    """
    Simulates a call to a Language Model (LLM) by generating a list of simulated
    responses from one or more rendered prompts.

    This class is a very loosely based on langchain_core.langchain_models.llms.BaseLLM
    """

    @property
    def call_log(self) -> List[SimCallLog]:
        return self._call_log

    @property
    def last_call_log(self) -> List[SimCallLog]:
        return self._last_call_log

    @property
    def all_call_tokens(self) -> int:
        return sum(x.response_tokens for x in self._call_log)

    @property
    def last_call_tokens(self) -> int:
        return sum(x.response_tokens for x in self._last_call_log)

    def _stats_by_prefix(
        self,
        logs: List[SimCallLog],
        order_by: StatSortType = "calls",
    ) -> List[StatsRecord]:
        """
        Calculate and return statistics grouped by prefix from a list of simulation call
        logs. Args:
            logs (List[SimCallLog]): A list of simulation call logs. order_by
            (StatSortType): The criterion to sort the statistics by.
                                     Can be "name", "calls", or "tokens".
        Returns:
            List[StatsRecord]: A list of statistics records sorted based on the
            specified criterion.
        Raises:
            ValueError: If the `order_by` value is not one of "name", "calls", or
            "tokens".
        """

        stats_by_prefix = {}

        for log in logs:
            if log.prefix not in stats_by_prefix:
                stats_by_prefix[log.prefix] = StatsRecord(
                    prefix=log.prefix,
                    total_count=0,
                    request_tokens=0,
                    response_tokens=0,
                    total_tokens=0,
                    success_count=0,
                    error_count=0,
                )

            stats_by_prefix[log.prefix].total_count += 1
            stats_by_prefix[log.prefix].request_tokens += log.request_tokens
            stats_by_prefix[log.prefix].response_tokens += log.response_tokens
            stats_by_prefix[log.prefix].total_tokens += (
                log.request_tokens + log.response_tokens
            )
            stats_by_prefix[log.prefix].success_count += log.status == "SUCCESS"
            stats_by_prefix[log.prefix].error_count += log.status == "ERROR"

        if order_by == "name":
            sorted_stats = sorted(stats_by_prefix.values(), key=lambda x: x.prefix)
        elif order_by == "calls":
            sorted_stats = sorted(
                stats_by_prefix.values(), key=lambda x: x.total_count, reverse=True
            )
        elif order_by == "tokens":
            sorted_stats = sorted(
                stats_by_prefix.values(),
                key=lambda x: x.request_tokens + x.response_tokens,
                reverse=True,
            )
        else:
            raise ValueError(f"Invalid order_by value: {order_by}")

        return sorted_stats

    def all_stats_by_prefix(
        self, order_by: StatSortType = "calls"
    ) -> List[StatsRecord]:
        """
        Retrieve all statistics records sorted by the specified order.

        Args:
            order_by (StatSortType): The criterion by which to sort the statistics records.

        Returns:
            List[StatsRecord]: A list of statistics records sorted according to the
            specified criterion.
        """
        return self._stats_by_prefix(self._call_log, order_by)

    def all_stats_by_prefix_json(
        self, order_by: StatSortType = "calls", **kwargs
    ) -> str:
        return json.dumps(
            [x.model_dump() for x in self.all_stats_by_prefix(order_by=order_by)],
            **kwargs,
        )

    def last_stats_by_prefix(
        self,
        order_by: StatSortType = "calls",
    ) -> List[StatsRecord]:
        """
        Retrieve the last statistics records sorted by the specified order.

        Args:
            order_by (StatSortType): The criterion to sort the statistics records.

        Returns:
            List[StatsRecord]: A list of statistics records sorted by the given criterion.
        """
        return self._stats_by_prefix(self._last_call_log, order_by)

    def last_stats_by_prefix_json(
        self,
        order_by: StatSortType = "calls",
        **kwargs,
    ) -> str:
        return json.dumps(
            [x.model_dump() for x in self.last_stats_by_prefix(order_by=order_by)],
            **kwargs,
        )

    def _cost_estimation(
        self, *, logs: List[SimCallLog], model_filter: List[str] | None = None
    ) -> List[ModelCostEstimation]:

        # Warning, since the simulator token counter relies on tiktoken, only OpenAI
        # models will be reasonably accurate. Google and Anthropic do not provide a
        # client-side tokenizer.

        estimated_costs: List[ModelCostEstimation] = []

        # Compute aggregate totals needed for price estimation
        sum_request_tokens = sum(
            x.request_tokens for x in logs if x.status == "SUCCESS"
        )
        sum_response_tokens = sum(
            x.response_tokens for x in logs if x.status == "SUCCESS"
        )
        sum_success_generations = len([x for x in logs if x.status == "SUCCESS"])

        for model_name, prices in _MODEL_PRICE_TABLE.items():

            if model_filter and model_name not in model_filter:
                continue

            request_per_1k = prices[0]
            response_per_1k = prices[1]

            request_cost = (
                Decimal(sum_request_tokens) / Decimal(1000)
            ) * request_per_1k
            response_cost = (
                Decimal(sum_response_tokens) / Decimal(1000)
            ) * response_per_1k
            total_cost = request_cost + response_cost

            # Round to 2 decimal places for monetary values
            request_cost = request_cost.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            response_cost = response_cost.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            total_cost = total_cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            estimated_costs.append(
                ModelCostEstimation(
                    llm_name=model_name,
                    request_tokens=sum_request_tokens,
                    response_tokens=sum_response_tokens,
                    total_tokens=sum_request_tokens + sum_response_tokens,
                    request_cost=float(request_cost),
                    response_cost=float(response_cost),
                    total_cost=float(total_cost),
                    generation_count=sum_success_generations,
                )
            )

        # Sort estimated costs by total cost in descending order
        estimated_costs = sorted(
            estimated_costs, key=lambda x: x.total_cost, reverse=True
        )

        return estimated_costs

    def all_cost_estimation(
        self, *, model_filter: List[str] | None = None
    ) -> List[ModelCostEstimation]:
        return self._cost_estimation(logs=self._call_log, model_filter=model_filter)

    def all_cost_estimation_json(
        self,
        *,
        model_filter: List[str] | None = None,
        **kwargs,
    ) -> str:

        return json.dumps(
            [
                x.model_dump()
                for x in self.all_cost_estimation(model_filter=model_filter)
            ],
            **kwargs,
        )

    def last_cost_estimation(
        self, *, model_filter: List[str] | None = None
    ) -> List[ModelCostEstimation]:
        return self._cost_estimation(
            logs=self._last_call_log, model_filter=model_filter
        )

    def last_cost_estimation_json(
        self, *, model_filter: List[str] | None = None, **kwargs
    ) -> str:
        return json.dumps(
            [
                x.model_dump()
                for x in self.last_cost_estimation(model_filter=model_filter)
            ],
            **kwargs,
        )

    def __init__(
        self,
        *,
        input_token_model: str = "gpt-4o",
        default_min_response_tokens: int = 100,
        default_max_response_tokens: int = 300,
        error_pct: float = 0.0,
        delay_pct: float = 0.0,
        delay_min_ms: int = 100,
        delay_max_ms: int = 1000,
        keep_stats: bool = True,
        **kwargs,
    ):
        self._lock = threading.Lock()
        self._last_call_log: List[SimCallLog] = []
        self._call_log: List[SimCallLog] = []

        # Initialize the input token model
        self.input_token_model = input_token_model
        self.tiktoken_enc = tiktoken.get_encoding(
            tiktoken.encoding_name_for_model(input_token_model)
        )

        # Set the default min/max response token lengths
        self.default_min_response_tokens = default_min_response_tokens
        self.default_max_response_tokens = default_max_response_tokens

        # Initialize the Faker data generator instance
        self.faker = Faker()

        # Error / delay configuration
        self.error_pct = error_pct
        self.delay_pct = delay_pct
        self.delay_min_ms = delay_min_ms
        self.delay_max_ms = delay_max_ms

        # Whether or not to save generation statistics, needed for summary logs and cost
        # estimation. If disabled it will speed up the simulator.
        self.keep_stats = keep_stats

        # Save raw init params for debugging
        self.init_kwargs = kwargs

    def _generate_text(
        self,
        *,
        append: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> Tuple[int, str]:

        gen_paragraphs: List[str] = []

        derived_min_tokens = min_tokens or self.default_min_response_tokens
        derived_max_tokens = max_tokens or self.default_max_response_tokens

        derived_append = [append] if append is not None else [""]

        if derived_min_tokens > derived_max_tokens:
            raise ValueError(
                f"min_tokens ({derived_min_tokens}) must be less than or equal to max_tokens ({derived_max_tokens})"
            )

        target_tokens = random.randint(derived_min_tokens, derived_max_tokens)

        latest_generation: str = ""
        latest_tokens: int = 0

        while latest_tokens < target_tokens:
            gen_paragraphs.append(self.faker.paragraph())

            latest_generation = "\n\n".join(gen_paragraphs + derived_append)
            latest_tokens = len(self.tiktoken_enc.encode(latest_generation))

        return (latest_tokens, latest_generation)

    def generate(
        self,
        prompt: str,
        *,
        n: int,
        prefix: str,
        append: str | None,
        min_tokens: int | None,
        max_tokens: int | None,
        **kwargs,
    ) -> SimCallLog:

        # Each generation has a chance to trigger a simulated delay
        if random.random() < self.delay_pct:

            delay_time_ms = random.uniform(self.delay_min_ms, self.delay_max_ms) / 1000
            LOG.debug(
                "Simulated LLM generation latency triggered for %.3fs", delay_time_ms
            )

            time.sleep(delay_time_ms)

        # Each geneation has a chance to trigger a simulated error
        if random.random() < self.error_pct:
            raise LLMSimulatorError("Simulated LLM generation error")

        gen_tokens, gen_text = self._generate_text(
            append=append, min_tokens=min_tokens, max_tokens=max_tokens
        )

        prompt_tokens = len(self.tiktoken_enc.encode(prompt))

        return SimCallLog(
            status="SUCCESS",
            prompt=prompt,
            request_tokens=prompt_tokens,
            response_tokens=gen_tokens,
            total_tokens=prompt_tokens + gen_tokens,
            text=gen_text,
            n=n,
            prefix=prefix,
            call_kwargs=kwargs,
        )

    def __call__(
        self,
        prompts: Union[List[str], str],
        *,
        n: int = 1,
        prefix: str = "GEN",
        append: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> List[str | None]:

        if isinstance(prompts, str):
            prompts = [prompts]

        # Simulate LLM responses for each prompt
        cur_call_log: List[SimCallLog] = []

        re_raise_sim_error: Union[LLMSimulatorError, None] = None

        for prompt in prompts:
            for n_index in range(n):

                # Generate a response
                try:
                    result = self.generate(
                        prompt=prompt,
                        n=n_index,
                        prefix=prefix,
                        append=append,
                        min_tokens=min_tokens,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                except LLMSimulatorError as ex:
                    # Save the error to re-raise later after we update the log entries,
                    # setting this will stop generating responses if an error occurs and
                    # only one prompt is provided, this somewhat simulates LangChain's
                    # behavior where batch inferencing will return errors raised in the
                    # batch vs pass the raised error to the caller for a single
                    # generation.
                    if len(prompts) == 1:
                        re_raise_sim_error = ex

                    # Map simulated error to a sensible log entry
                    prompt_tokens = len(self.tiktoken_enc.encode(prompt))
                    result = SimCallLog(
                        status="ERROR",
                        prompt=prompt,
                        request_tokens=prompt_tokens,
                        response_tokens=0,
                        total_tokens=prompt_tokens,
                        text=None,
                        n=n_index,
                        prefix=prefix,
                        call_kwargs=kwargs,
                    )

                # Append instance call log entry
                cur_call_log.append(result)

                if re_raise_sim_error is not None:
                    break

        # Append the current call log to the global call log
        if self.keep_stats:
            with self._lock:
                self._call_log.extend(cur_call_log)
                self._last_call_log = cur_call_log

        if re_raise_sim_error is not None:
            raise re_raise_sim_error

        return [x.text for x in cur_call_log]
