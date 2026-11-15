"""Regression checks for input validation, API presentation and evaluation."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_contract import parse_inputs, percent
from evaluate_models import evaluate
import app as web


class ContractTests(unittest.TestCase):
    def test_rejects_missing_nonfinite_and_negative_features(self):
        valid = {str(i): '1' for i in range(1, 9)}
        self.assertEqual(len(parse_inputs(valid)), 8)
        for name, value in [('3', '-2'), ('4', 'nan'), ('6', 'inf'), ('8', '0')]:
            with self.subTest(field=name), self.assertRaises(ValueError):
                parse_inputs({**valid, name: value})
        with self.assertRaisesRegex(ValueError, 'Glucose'):
            parse_inputs({key: value for key, value in valid.items() if key != '2'})

    def test_probabilities_must_be_bounded(self):
        self.assertEqual(percent(0.215), '21.5%')
        for val in (-0.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                percent(val)


class FakeModel:
    def predict_proba(self, features):
        return [[0.8, 0.2]]


class FakeScaler:
    def transform(self, frame):
        return frame


class ApiTests(unittest.TestCase):
    def test_health_and_form_validation(self):
        client = web.app.test_client()
        self.assertEqual(client.get('/health').json['clinical_use'], False)
        self.assertEqual(client.get('/predict').status_code, 405)
        self.assertEqual(client.post('/predict', data={'1': '1'}).status_code, 400)

    def test_template_does_not_choose_maximum_as_best(self):
        stub = {'scaler': FakeScaler(), **{name: FakeModel() for name in (
            'KNN', 'XGBoost', 'Logistic Regression', 'Random Forest')}}
        with patch.object(web, 'get_models', return_value=stub):
            response = web.app.test_client().post(
                '/predict', data={str(i): '1' for i in range(1, 9)}
            )
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('20.0%', html)
        self.assertIn('Not a diagnosis', html)
        self.assertNotIn('best model predicts', html.lower())


class EvaluationTests(unittest.TestCase):
    def test_pipeline_runs_on_untransformed_source(self):
        report = evaluate(ROOT / 'diabetes.csv')
        self.assertEqual(report['rows'], 768)
        self.assertEqual(report['train_rows'] + report['test_rows'], 768)
        self.assertEqual(report['models'][0]['model'], 'Most frequent baseline')
        for model in report['models']:
            for metric in ('accuracy', 'f1', 'recall', 'roc_auc'):
                self.assertGreaterEqual(model[metric], 0)
                self.assertLessEqual(model[metric], 1)


if __name__ == '__main__':
    unittest.main()
