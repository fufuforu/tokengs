# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .input_types import (
    EncoderLatent,
    ModelInput,
    ModelInputDecoder,
    ModelInputEncoder,
    ModelSupervision,
    Reconstruction,
    split_data,
)
from .tokengs import TokenGS
from .prompt_matching import (
    PromptConditionedTokenMatcher,
    PromptEncoder,
    PromptGaussianDecoder,
)
from .prompt_tokengs import PromptTokenGS
from .semantic_adapter_v2 import (
    C3G8_CLASS_NAMES,
    PromptSemanticAdapter,
    SemanticMatcherV2,
    SemanticTokenAdapter,
)
from .semantic_tokengs_v2 import SemanticTokenGSv2
from .semantic_tokengs_v3 import SemanticTokenGSv3
from .semantic_tokengs_v4 import SemanticTokenGSv4
from .semantic_tokengs_v5 import SemanticTokenGSv5
from .semantic_tokengs_v6 import SemanticTokenGSv6
from .conditional_prompt_tokengs import ConditionalPromptTokenGS

# Model registry
model_registry = {
    'tokengs': TokenGS,
    'prompt_tokengs': PromptTokenGS,
    'semantic_tokengs_v2': SemanticTokenGSv2,
    'semantic_tokengs_v3': SemanticTokenGSv3,
    'semantic_tokengs_v4': SemanticTokenGSv4,
    'semantic_tokengs_v5': SemanticTokenGSv5,
    'semantic_tokengs_v6': SemanticTokenGSv6,
    'conditional_prompt_tokengs': ConditionalPromptTokenGS,
}

# Export for convenience
__all__ = [
    'TokenGS',
    'split_data',
    'ModelInput',
    'ModelInputEncoder',
    'ModelInputDecoder',
    'ModelSupervision',
    'Reconstruction',
    'EncoderLatent',
    'PromptEncoder',
    'PromptGaussianDecoder',
    'PromptConditionedTokenMatcher',
    'PromptTokenGS',
    'C3G8_CLASS_NAMES',
    'SemanticTokenAdapter',
    'PromptSemanticAdapter',
    'SemanticMatcherV2',
    'SemanticTokenGSv2',
    'SemanticTokenGSv4',
    'SemanticTokenGSv5',
    'SemanticTokenGSv6',
    'ConditionalPromptTokenGS',
    'model_registry',
]
