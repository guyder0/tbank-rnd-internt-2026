"""Offline adapter checks; pretrained weights are tested separately on the host."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from research.generative_recsys.src.tokenizer import content_embeddings, validate_prepared_config


class SentenceT5Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('research/generative_recsys/configs/default.json').read_text())
        self.config.update(embedding_backend='sentence_t5',embedding_model='sentence-transformers/sentence-t5-base')

    def test_full_pipeline_receives_raw_russian_text_and_encoding_options(self):
        from unittest.mock import MagicMock
        factory = MagicMock()
        model = factory.return_value
        model.float.return_value = model
        model.encode.return_value = np.eye(2, 768, dtype='float32')
        model.__getitem__.return_value.auto_model.config._commit_hash = 'test-revision'
        texts = ['Название: Фильм', 'Жанр: драма']
        with patch.dict('sys.modules', {'sentence_transformers':SimpleNamespace(SentenceTransformer=factory)}):
            vectors, metadata = content_embeddings(texts, self.config, 'cpu')
        self.assertEqual(vectors.shape, (2, 768))
        self.assertEqual(model.encode.call_args.args[0], texts)
        self.assertTrue(model.encode.call_args.kwargs['normalize_embeddings'])
        self.assertEqual(model.encode.call_args.kwargs['device'], 'cpu')
        self.assertEqual(model.max_seq_length, 256)
        self.assertEqual(metadata['resolved_revision'], 'test-revision')
        self.assertFalse(factory.call_args.kwargs['trust_remote_code'])

    def test_wrong_dimension_rejected(self):
        from unittest.mock import MagicMock
        factory = MagicMock()
        factory.return_value.encode.return_value = np.ones((1, 96))
        with patch.dict('sys.modules', {'sentence_transformers':SimpleNamespace(SentenceTransformer=factory)}):
            with self.assertRaisesRegex(ValueError, 'Unexpected Sentence-T5'):
                content_embeddings(['фильм'], self.config)

    def test_old_embeddings_and_changed_truncation_cannot_reuse_cache(self):
        for changes in [{'embedding_backend':'tfidf'}, {'embedding_max_length':128}]:
            with self.assertRaises(ValueError):
                validate_prepared_config(self.config, {**self.config, **changes})


class LocalT5IntegrationTests(unittest.TestCase):
    def test_real_tiny_t5_pipeline_without_download(self):
        import importlib.util
        if not importlib.util.find_spec('sentence_transformers'):
            self.skipTest('Sentence-Transformers not installed')
        import tempfile
        import subprocess
        import sys
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import T5Config, T5EncoderModel, PreTrainedTokenizerFast
        from sentence_transformers import SentenceTransformer, models
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root/'base'; base.mkdir()
            tokenizer = Tokenizer(WordLevel({'[PAD]':0,'[UNK]':1,'фильм':2,'драма':3}, unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]',
                unk_token='[UNK]', model_input_names=['input_ids','attention_mask'])
            tokenizer.save_pretrained(base)
            T5EncoderModel(T5Config(vocab_size=4,d_model=16,d_ff=32,num_layers=1,
                num_heads=2,d_kv=8)).save_pretrained(base)
            model = SentenceTransformer(modules=[models.Transformer(str(base)),
                models.Pooling(16),models.Dense(16,8)],device='cpu')
            checkpoint = root/'sentence_t5'; model.save(str(checkpoint))
            config = json.loads(Path('research/generative_recsys/configs/default.json').read_text())
            config.update(embedding_backend='sentence_t5',embedding_model=str(checkpoint),embedding_dim=8)
            vectors, _ = content_embeddings(['фильм','фильм драма'], config)
            self.assertEqual(vectors.shape,(2,8))
            np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
            config_file = root/'config.json'; config_file.write_text(json.dumps(config))
            result = subprocess.run([sys.executable,'-m','research.generative_recsys.run',
                '--smoke','--pretrained-smoke','--config',str(config_file),'--output',str(root/'run')],
                capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(json.loads((root/'run/status.json').read_text())['status'],'completed')
