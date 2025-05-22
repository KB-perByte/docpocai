#!/usr/bin/env python3
"""
Ansible Network Module Test Generator

This tool uses machine learning to analyze Ansible module documentation 
and generate appropriate unit tests automatically.
"""

import os
import re
import sys
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import requests
from bs4 import BeautifulSoup
import ast
import importlib.util
from dataclasses import dataclass

# For ML components
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Optional: if available
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

@dataclass
class ModuleParam:
    """Represents a single module parameter from documentation"""
    name: str
    type: str
    required: bool
    description: str
    default: Any = None
    choices: List[str] = None

@dataclass
class ModuleInfo:
    """Contains the parsed module information from docs"""
    name: str
    description: str
    params: List[ModuleParam]
    examples: List[str]
    return_values: Dict[str, Any]


class DocParser:
    """Parse Ansible module documentation in RST format"""
    
    def __init__(self, doc_path: str):
        self.doc_path = doc_path
        self.raw_content = self._read_content()
        
    def _read_content(self) -> str:
        """Read the raw documentation content"""
        with open(self.doc_path, 'r') as f:
            return f.read()
            
    def extract_module_info(self) -> ModuleInfo:
        """Extract module information from RST documentation"""
        # Extract module name
        module_name_match = re.search(r'\.\. _(\w+):', self.raw_content)
        if module_name_match:
            module_name = module_name_match.group(1)
        else:
            module_name = os.path.basename(self.doc_path).replace('.rst', '')
        
        # Extract description
        description = ""
        desc_match = re.search(r'Description\n-+\n\n(.*?)(\n\n|$)', self.raw_content, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()
        
        # Extract parameters
        params = self._extract_parameters()
        
        # Extract examples
        examples = self._extract_examples()
        
        # Extract return values
        return_values = self._extract_return_values()
        
        return ModuleInfo(
            name=module_name,
            description=description,
            params=params,
            examples=examples,
            return_values=return_values
        )
    
    def _extract_parameters(self) -> List[ModuleParam]:
        """Extract parameter details from the documentation"""
        params = []
        # Find the Parameters section
        params_section = re.search(r'Parameters\n-+\n\n(.*?)(\n\n\w+\n-+|$)', 
                                  self.raw_content, re.DOTALL)
        
        if not params_section:
            return params
            
        param_content = params_section.group(1)
        
        # Extract individual parameters
        param_blocks = re.findall(r'(\w+)\n(\s+.*?)(?=\n\w+\n|\Z)', param_content, re.DOTALL)
        
        for name, details in param_blocks:
            # Extract type
            type_match = re.search(r'[Tt]ype:\s*([\w\[\]]+)', details)
            param_type = type_match.group(1) if type_match else "str"
            
            # Extract required status
            required = "required" in details.lower()
            
            # Extract description
            desc_lines = [line.strip() for line in details.split('\n') if line.strip()]
            description = ' '.join(desc_lines)
            
            # Extract choices if available
            choices = None
            choices_match = re.search(r'[Cc]hoices:\s*\[(.*?)\]', details)
            if choices_match:
                choices_str = choices_match.group(1)
                choices = [c.strip().strip('"\'') for c in choices_str.split(',')]
            
            # Extract default if available
            default = None
            default_match = re.search(r'[Dd]efault:\s*([\w\'\"]+)', details)
            if default_match:
                default = default_match.group(1)
                # Convert string representation to appropriate Python type
                if default.lower() == 'none':
                    default = None
                elif default.lower() == 'true':
                    default = True
                elif default.lower() == 'false':
                    default = False
                elif default.isdigit():
                    default = int(default)
                else:
                    # Strip quotes if present
                    default = default.strip('\'"')
            
            param = ModuleParam(
                name=name.strip(),
                type=param_type,
                required=required,
                description=description,
                default=default,
                choices=choices
            )
            params.append(param)
        
        return params
    
    def _extract_examples(self) -> List[str]:
        """Extract example configurations from documentation"""
        examples = []
        example_section = re.search(r'Examples\n-+\n\n(.*?)(\n\n\w+\n-+|$)', 
                                  self.raw_content, re.DOTALL)
        
        if not example_section:
            return examples
            
        example_content = example_section.group(1)
        
        # Find YAML code blocks
        yaml_blocks = re.findall(r'.. code-block:: yaml\n\n((?:\s+.*\n)+)', example_content)
        
        for block in yaml_blocks:
            # Clean up indentation
            clean_block = '\n'.join([line.strip() for line in block.split('\n')])
            examples.append(clean_block)
        
        return examples
    
    def _extract_return_values(self) -> Dict[str, Any]:
        """Extract return value definitions from documentation"""
        return_values = {}
        
        return_section = re.search(r'Return Values\n-+\n\n(.*?)(\n\n\w+\n-+|$)', 
                                   self.raw_content, re.DOTALL)
        
        if not return_section:
            return return_values
            
        return_content = return_section.group(1)
        
        # This is a simplified parser - more complex parsing may be needed
        # for nested return values
        return_blocks = re.findall(r'(\w+)\n(\s+.*?)(?=\n\w+\n|\Z)', return_content, re.DOTALL)
        
        for name, details in return_blocks:
            return_values[name] = {
                'description': details.strip(),
                'returned': 'always',  # Default, would need more parsing for accurate value
                'type': 'dict'  # Default, would need more parsing for accurate type
            }
        
        return return_values


class TestTemplateGenerator:
    """Generate Ansible unit test templates from module information"""
    
    def __init__(self, module_info: ModuleInfo):
        self.module_info = module_info
        
    def generate_basic_test(self) -> str:
        """Generate a basic unit test structure for the module"""
        module_name = self.module_info.name
        
        test_content = f"""
# Copyright 2021 Red Hat
# GNU General Public License v3.0+
# (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

from __future__ import absolute_import, division, print_function
__metaclass__ = type

import unittest
from unittest.mock import patch
from ansible_collections.cisco.iosxr.plugins.modules import {module_name}
from ansible_collections.cisco.iosxr.tests.unit.modules.utils import set_module_args
from ansible_collections.cisco.iosxr.tests.unit.modules.utils import AnsibleFailJson
from ansible_collections.cisco.iosxr.tests.unit.modules.utils import AnsibleExitJson

class Test{module_name.title()}Module(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
        
    def test_module_arguments(self):
        \"\"\"Test valid module arguments\"\"\"
        set_module_args({{
{self._generate_module_args_sample()}
        }})
        with self.assertRaises(AnsibleExitJson) as result:
            {module_name}.main()
            
    def test_module_fail_when_required_args_missing(self):
        \"\"\"Test failure when required arguments are missing\"\"\"
        with self.assertRaises(AnsibleFailJson) as result:
            set_module_args({{}})
            {module_name}.main()
            
{self._generate_specific_tests()}
"""
        return test_content
    
    def _generate_module_args_sample(self, indent=12) -> str:
        """Generate sample module arguments based on parameters"""
        args = []
        for param in self.module_info.params:
            # Skip params with default values that aren't required
            if not param.required and param.default is None:
                continue
                
            # Generate appropriate value based on type
            if param.choices:
                value = param.choices[0]  # Pick first choice
            elif param.type == 'str':
                value = f"'test_{param.name}'"
            elif param.type == 'int':
                value = '123'
            elif param.type == 'bool':
                value = 'True'
            elif param.type == 'list':
                value = "['test_item']"
            elif param.type in ['dict', 'dictionary']:
                value = "{'key': 'value'}"
            else:
                value = "'test_value'"
                
            # Add the argument
            args.append(f"'{param.name}': {value}")
        
        return ',\n'.join([' ' * indent + arg for arg in args])
    
    def _generate_specific_tests(self) -> str:
        """Generate specific test methods based on module functionality"""
        specific_tests = []
        
        # Always add a mock test for module execution
        mock_test = f"""    @patch('{self.module_info.name}.run_commands')
    def test_module_execution(self):
        \"\"\"Test module execution with mocked run_commands\"\"\"
        set_module_args({{
{self._generate_module_args_sample()}
        }})
        
        # Mock the run_commands response
        commands_output = [
            # Add expected command outputs here
            '''show running-config | include logging'''
        ]
        
        with patch.object({self.module_info.name}, 'run_commands', return_value=commands_output) as mock_run_commands:
            with self.assertRaises(AnsibleExitJson) as result:
                {self.module_info.name}.main()
                
            # Verify that run_commands was called with appropriate arguments
            mock_run_commands.assert_called_once()
            # Add more assertions about the result as needed
"""
        specific_tests.append(mock_test)
        
        # Check if this is a state-based module (merged, replaced, etc.)
        if any(param.name == 'state' for param in self.module_info.params):
            state_param = next(param for param in self.module_info.params if param.name == 'state')
            if state_param.choices:
                for state in state_param.choices:
                    state_test = self._generate_state_test(state)
                    specific_tests.append(state_test)
        
        return '\n'.join(specific_tests)
    
    def _generate_state_test(self, state: str) -> str:
        """Generate a test for a specific state"""
        state_test = f"""    @patch('{self.module_info.name}.run_commands')
    def test_{state}_state(self):
        \"\"\"Test module with '{state}' state\"\"\"
        set_module_args({{
{self._generate_module_args_sample(indent=12)},
            'state': '{state}'
        }})
        
        commands_output = [
            # Add expected command outputs for {state} state
            '''show running-config | include logging'''
        ]
        
        with patch.object({self.module_info.name}, 'run_commands', return_value=commands_output) as mock_run_commands:
            with self.assertRaises(AnsibleExitJson) as result:
                {self.module_info.name}.main()
            
            # Add assertions specific to {state} state
"""
        return state_test


class MLTestGenerator:
    """Machine learning based test generator for Ansible modules"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.load_model()
        
    def load_model(self):
        """Load a pre-trained model if available"""
        if not self.model_path or not os.path.exists(self.model_path):
            # Initialize default model components if no saved model
            self.vectorizer = TfidfVectorizer(max_features=1000)
            # Simple model for demonstration - would be replaced with a more sophisticated model
            return
            
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data.get('model')
            self.vectorizer = model_data.get('vectorizer')
        except Exception as e:
            print(f"Error loading model: {e}")
            # Initialize default models
            self.vectorizer = TfidfVectorizer(max_features=1000)
    
    def train(self, docs_path: str, tests_path: str):
        """Train the model on documentation and existing tests"""
        # This is a simplified training process - a real implementation would be more complex
        
        # 1. Collect documentation and test pairs
        doc_test_pairs = self._collect_doc_test_pairs(docs_path, tests_path)
        
        # 2. Extract features from documentation
        doc_contents = [pair[0] for pair in doc_test_pairs]
        self.vectorizer = TfidfVectorizer(max_features=1000)
        doc_features = self.vectorizer.fit_transform(doc_contents)
        
        # 3. Extract features from tests
        test_contents = [pair[1] for pair in doc_test_pairs]
        
        # 4. Train a model to predict test content from doc features
        # This is where you'd implement your ML model
        # For this demonstration, we'll just save the vectorizer
        
        # 5. Save the model
        model_data = {
            'vectorizer': self.vectorizer,
            'model': None  # Replace with actual model
        }
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model_data, self.model_path)
        
    def _collect_doc_test_pairs(self, docs_path: str, tests_path: str) -> List[Tuple[str, str]]:
        """Collect paired documentation and test files"""
        doc_test_pairs = []
        
        # Find all RST documentation files
        doc_files = list(Path(docs_path).glob('**/*.rst'))
        
        for doc_file in doc_files:
            # Extract module name from filename
            module_name = doc_file.stem
            
            # Look for corresponding test file
            potential_test_paths = [
                Path(tests_path) / 'unit' / 'modules' / f'test_{module_name}.py',
                Path(tests_path) / 'unit' / f'test_{module_name}.py',
                Path(tests_path) / f'test_{module_name}.py'
            ]
            
            test_file = None
            for test_path in potential_test_paths:
                if test_path.exists():
                    test_file = test_path
                    break
                    
            if test_file:
                # Read content
                with open(doc_file, 'r') as f:
                    doc_content = f.read()
                with open(test_file, 'r') as f:
                    test_content = f.read()
                    
                doc_test_pairs.append((doc_content, test_content))
                
        return doc_test_pairs
        
    def generate_advanced_test(self, module_info: ModuleInfo) -> str:
        """Generate tests using ML model insights"""
        # This is where the trained model would be used to generate
        # more sophisticated tests based on patterns learned
        
        # For this demonstration, we'll use the template generator
        # and enhance it with some heuristics that an ML model might learn
        
        template_gen = TestTemplateGenerator(module_info)
        basic_test = template_gen.generate_basic_test()
        
        # Enhance with additional test cases based on module type
        enhanced_test = basic_test
        
        # Example of a heuristic: Add config idempotency test if this is a config module
        if "config" in module_info.name or any("config" in p.name for p in module_info.params):
            idempotency_test = self._generate_idempotency_test(module_info)
            enhanced_test = enhanced_test + "\n" + idempotency_test
            
        # Add test for each example in the documentation
        if module_info.examples:
            example_tests = self._generate_tests_from_examples(module_info)
            enhanced_test = enhanced_test + "\n" + example_tests
            
        return enhanced_test
        
    def _generate_idempotency_test(self, module_info: ModuleInfo) -> str:
        """Generate an idempotency test"""
        return f"""    @patch('{module_info.name}.run_commands')
    def test_idempotency(self):
        \"\"\"Test module idempotency (running twice should not change result)\"\"\"
        set_module_args({{
{TestTemplateGenerator(module_info)._generate_module_args_sample()}
        }})
        
        # First run
        commands_output = [
            '''show running-config | include logging'''
        ]
        
        with patch.object({module_info.name}, 'run_commands', return_value=commands_output) as mock_run_commands:
            with self.assertRaises(AnsibleExitJson) as result1:
                {module_info.name}.main()
            result1_changed = result1.exception.args[0]['changed']
            
        # Second run should not result in changes
        with patch.object({module_info.name}, 'run_commands', return_value=commands_output) as mock_run_commands:
            with self.assertRaises(AnsibleExitJson) as result2:
                {module_info.name}.main()
            result2_changed = result2.exception.args[0]['changed']
            
        # Second run should report no changes
        self.assertTrue(result1_changed)  # First run should report changes
        self.assertFalse(result2_changed)  # Second run should report no changes
"""
        
    def _generate_tests_from_examples(self, module_info: ModuleInfo) -> str:
        """Generate tests based on documentation examples"""
        example_tests = []
        
        for i, example in enumerate(module_info.examples):
            # Try to extract a task dict from the example YAML
            try:
                example_yaml = yaml.safe_load(example)
                
                # Find the task that uses this module
                module_task = None
                if isinstance(example_yaml, list):
                    for task in example_yaml:
                        if module_info.name in task:
                            module_task = task
                            break
                elif isinstance(example_yaml, dict) and module_info.name in example_yaml:
                    module_task = example_yaml
                    
                if module_task:
                    # Extract the module arguments
                    module_args = module_task.get(module_info.name, {})
                    
                    # Create a test using these arguments
                    example_test = f"""    @patch('{module_info.name}.run_commands')
    def test_example_{i+1}(self):
        \"\"\"Test using example {i+1} from documentation\"\"\"
        set_module_args({module_args})
        
        commands_output = [
            '''show running-config | include logging'''
        ]
        
        with patch.object({module_info.name}, 'run_commands', return_value=commands_output) as mock_run_commands:
            with self.assertRaises(AnsibleExitJson) as result:
                {module_info.name}.main()
                
            # Add assertions specific to this example
"""
                    example_tests.append(example_test)
            except:
                # If parsing fails, skip this example
                continue
                
        return '\n'.join(example_tests)


class TestGenerator:
    """Main class to coordinate test generation"""
    
    def __init__(self, doc_path: str, output_path: str, model_path: Optional[str] = None):
        self.doc_path = doc_path
        self.output_path = output_path
        self.model_path = model_path
        
    def generate_test(self) -> str:
        """Generate a test file based on the documentation"""
        # 1. Parse the documentation
        parser = DocParser(self.doc_path)
        module_info = parser.extract_module_info()
        
        # 2. If ML model available, use it for advanced test generation
        if self.model_path:
            ml_generator = MLTestGenerator(self.model_path)
            test_content = ml_generator.generate_advanced_test(module_info)
        else:
            # Otherwise use template-based generation
            template_gen = TestTemplateGenerator(module_info)
            test_content = template_gen.generate_basic_test()
        
        # 3. Write the test to the output path
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, 'w') as f:
                f.write(test_content)
            
        return test_content


class ModelTrainer:
    """Class to train an ML model on Ansible module documentation and tests"""
    
    def __init__(self, ansible_repo_path: str, output_model_path: str):
        self.ansible_repo_path = ansible_repo_path
        self.output_model_path = output_model_path
        
    def train(self):
        """Train and save the model"""
        # Initialize ML generator
        ml_generator = MLTestGenerator(self.output_model_path)
        
        # Find docs and tests paths
        collections_path = os.path.join(self.ansible_repo_path, 'collections')
        
        # Get all collections
        collections = []
        for namespace_dir in os.listdir(collections_path):
            namespace_path = os.path.join(collections_path, namespace_dir)
            if os.path.isdir(namespace_path):
                for collection_dir in os.listdir(namespace_path):
                    collection_path = os.path.join(namespace_path, collection_dir)
                    if os.path.isdir(collection_path):
                        collections.append((namespace_dir, collection_dir, collection_path))
        
        # Process each collection
        for namespace, collection, path in collections:
            docs_path = os.path.join(path, 'docs')
            tests_path = os.path.join(path, 'tests')
            
            if os.path.exists(docs_path) and os.path.exists(tests_path):
                print(f"Training on collection: {namespace}.{collection}")
                ml_generator.train(docs_path, tests_path)
                
        print(f"Model trained and saved to {self.output_model_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Ansible module tests from documentation')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate test command
    gen_parser = subparsers.add_parser('generate', help='Generate a test file')
    gen_parser.add_argument('--doc', required=True, help='Path to the module documentation RST file')
    gen_parser.add_argument('--output', required=True, help='Path to save the generated test file')
    gen_parser.add_argument('--model', help='Path to trained model file (optional)')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train the ML model')
    train_parser.add_argument('--repo', required=True, help='Path to Ansible collections repository')
    train_parser.add_argument('--output', required=True, help='Path to save the trained model')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generator = TestGenerator(args.doc, args.output, args.model)
        test_content = generator.generate_test()
        print(f"Generated test saved to {args.output}")
        
    elif args.command == 'train':
        trainer = ModelTrainer(args.repo, args.output)
        trainer.train()
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
