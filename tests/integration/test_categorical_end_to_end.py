"""End-to-end integration tests for categorical feature support."""

import tempfile

import torch

from arc.graph.model import build_model_from_yaml
from arc.ml.artifacts import ModelArtifact, ModelArtifactManager
from arc.ml.data import DataProcessor


class TestCategoricalEndToEnd:
    """Test complete categorical feature workflow."""

    def test_single_categorical_feature_workflow(self):
        """Test full workflow with a single categorical feature.

        Workflow:
        1. Create sample data with categorical column
        2. Fit label encoder and transform categorical data
        3. Build model with automatic embedding layer
        4. Train model briefly
        5. Save model with vocabularies
        6. Load model and vocabularies
        7. Make predictions on new data
        """
        # Step 1: Create sample data with categorical feature
        categories = ["red", "green", "blue", "red", "green", "blue", "red", "green"]
        numerical_features = torch.randn(8, 5)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)

        # Step 2: Fit label encoder
        processor = DataProcessor()
        fit_result = processor._execute_operator("fit.label_encoder", {"x": categories})
        vocabulary_state = fit_result["state"]

        # Transform categorical data to indices
        transform_result = processor._execute_operator(
            "transform.label_encode",
            {"x": categories, "state": vocabulary_state},
        )
        categorical_indices = transform_result["output"]

        # Step 3: Build model with automatic embedding
        yaml_spec = """
        inputs:
          category:
            dtype: long
            shape: [null]
            categorical: true
            embedding_dim: 8
            vocab_size: 3  # red, green, blue

          features:
            dtype: float32
            shape: [null, 5]

        graph:
          - name: concat
            type: torch.cat
            params:
              dim: 1
            inputs: [category, features]

          - name: hidden
            type: torch.nn.Linear
            params:
              in_features: 13  # 8 (embedding) + 5 (features)
              out_features: 16
            inputs:
              input: concat

          - name: activation
            type: torch.nn.functional.relu
            inputs: [hidden.output]

          - name: output
            type: torch.nn.Linear
            params:
              in_features: 16
              out_features: 1
            inputs:
              input: activation

        outputs:
          prediction: output.output
        """

        model = build_model_from_yaml(yaml_spec)

        # Verify embedding was created
        assert "category" in model.embeddings
        assert model.embeddings["category"].num_embeddings == 3
        assert model.embeddings["category"].embedding_dim == 8

        # Step 4: Train model briefly (just verify it works)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(5):  # Just a few iterations
            optimizer.zero_grad()
            predictions = model(
                category=categorical_indices, features=numerical_features
            )
            loss = loss_fn(predictions.squeeze(), targets)
            loss.backward()
            optimizer.step()

        # Verify training worked
        assert loss.item() < 1.0  # Loss should be reasonable

        # Step 5: Save model with vocabularies
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ModelArtifactManager(tmp_dir)

            vocabularies = {"category": vocabulary_state}

            artifact = ModelArtifact(
                model_id="categorical_model",
                model_name="Categorical Test Model",
                version=1,
            )

            artifact_dir = manager.save_model_artifact(
                model=model,
                artifact=artifact,
                vocabularies=vocabularies,
            )

            # Verify files were created
            assert (artifact_dir / "model_state.pt").exists()
            assert (artifact_dir / "vocabularies.json").exists()

            # Step 6: Load model and vocabularies
            loaded_vocabs = manager.load_vocabularies("categorical_model", version=1)
            assert loaded_vocabs is not None
            assert "category" in loaded_vocabs
            assert loaded_vocabs["category"]["vocab_size"] == 3

            # Rebuild model
            new_model = build_model_from_yaml(yaml_spec)
            state_dict, _ = manager.load_model_state_dict(
                "categorical_model", version=1
            )
            new_model.load_state_dict(state_dict)

            # Step 7: Make predictions on new data
            new_model.eval()
            new_categories = ["blue", "red", "green"]

            # Transform new categories using loaded vocabulary
            new_transform = processor._execute_operator(
                "transform.label_encode",
                {"x": new_categories, "state": loaded_vocabs["category"]},
            )
            new_categorical_indices = new_transform["output"]

            new_features = torch.randn(3, 5)

            with torch.no_grad():
                predictions = new_model(
                    category=new_categorical_indices, features=new_features
                )

            # Verify predictions have correct shape
            assert predictions.shape == (3, 1)

    def test_multiple_categorical_features_workflow(self):
        """Test workflow with multiple categorical features."""
        # Sample data
        colors = ["red", "blue", "red", "green", "blue", "green"]
        sizes = ["small", "large", "medium", "small", "large", "medium"]
        numerical = torch.randn(6, 4)
        targets = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32)

        # Fit encoders for both categorical features
        processor = DataProcessor()

        color_fit = processor._execute_operator("fit.label_encoder", {"x": colors})
        color_vocab = color_fit["state"]

        size_fit = processor._execute_operator("fit.label_encoder", {"x": sizes})
        size_vocab = size_fit["state"]

        # Transform both
        color_indices = processor._execute_operator(
            "transform.label_encode", {"x": colors, "state": color_vocab}
        )["output"]

        size_indices = processor._execute_operator(
            "transform.label_encode", {"x": sizes, "state": size_vocab}
        )["output"]

        # Build model with two categorical inputs
        yaml_spec = """
        inputs:
          color:
            dtype: long
            shape: [null]
            categorical: true
            embedding_dim: 4
            vocab_size: 3

          size:
            dtype: long
            shape: [null]
            categorical: true
            embedding_dim: 3
            vocab_size: 3

          features:
            dtype: float32
            shape: [null, 4]

        graph:
          - name: concat_all
            type: torch.cat
            params:
              dim: 1
            inputs: [color, size, features]

          - name: mlp
            type: torch.nn.Linear
            params:
              in_features: 11  # 4 + 3 + 4
              out_features: 1
            inputs:
              input: concat_all

        outputs:
          prediction: mlp.output
        """

        model = build_model_from_yaml(yaml_spec)

        # Verify both embeddings created
        assert len(model.embeddings) == 2
        assert "color" in model.embeddings
        assert "size" in model.embeddings

        # Train briefly
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            preds = model(color=color_indices, size=size_indices, features=numerical)
            loss = loss_fn(preds.squeeze(), targets)
            loss.backward()
            optimizer.step()

        # Save with both vocabularies
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ModelArtifactManager(tmp_dir)

            vocabularies = {
                "color": color_vocab,
                "size": size_vocab,
            }

            artifact = ModelArtifact(
                model_id="multi_cat_model",
                model_name="Multi Categorical Model",
                version=1,
            )

            manager.save_model_artifact(
                model=model, artifact=artifact, vocabularies=vocabularies
            )

            # Load and verify
            loaded_vocabs = manager.load_vocabularies("multi_cat_model")
            assert len(loaded_vocabs) == 2
            assert "color" in loaded_vocabs
            assert "size" in loaded_vocabs

            # Make predictions with new data
            new_model = build_model_from_yaml(yaml_spec)
            state_dict, _ = manager.load_model_state_dict("multi_cat_model")
            new_model.load_state_dict(state_dict)

            new_colors = ["green", "red"]
            new_sizes = ["large", "small"]

            new_color_idx = processor._execute_operator(
                "transform.label_encode",
                {"x": new_colors, "state": loaded_vocabs["color"]},
            )["output"]

            new_size_idx = processor._execute_operator(
                "transform.label_encode",
                {"x": new_sizes, "state": loaded_vocabs["size"]},
            )["output"]

            new_features = torch.randn(2, 4)

            new_model.eval()
            with torch.no_grad():
                preds = new_model(
                    color=new_color_idx, size=new_size_idx, features=new_features
                )

            assert preds.shape == (2, 1)

    def test_categorical_only_workflow(self):
        """Test workflow with only categorical features (no numerical)."""
        # Sample categorical data
        user_ids = ["user_1", "user_2", "user_3", "user_1", "user_2"]
        item_ids = ["item_a", "item_b", "item_a", "item_c", "item_b"]
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])

        # Encode
        processor = DataProcessor()

        user_fit = processor._execute_operator("fit.label_encoder", {"x": user_ids})
        user_vocab = user_fit["state"]

        item_fit = processor._execute_operator("fit.label_encoder", {"x": item_ids})
        item_vocab = item_fit["state"]

        user_idx = processor._execute_operator(
            "transform.label_encode", {"x": user_ids, "state": user_vocab}
        )["output"]

        item_idx = processor._execute_operator(
            "transform.label_encode", {"x": item_ids, "state": item_vocab}
        )["output"]

        # Model with only categorical inputs
        yaml_spec = """
        inputs:
          user:
            dtype: long
            shape: [null]
            categorical: true
            embedding_dim: 8
            vocab_size: 3

          item:
            dtype: long
            shape: [null]
            categorical: true
            embedding_dim: 8
            vocab_size: 3

        graph:
          - name: concat
            type: torch.cat
            params:
              dim: 1
            inputs: [user, item]

          - name: classifier
            type: torch.nn.Linear
            params:
              in_features: 16
              out_features: 1
            inputs:
              input: concat

        outputs:
          score: classifier.output
        """

        model = build_model_from_yaml(yaml_spec)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            scores = model(user=user_idx, item=item_idx)
            loss = loss_fn(scores.squeeze(), labels)
            loss.backward()
            optimizer.step()

        # Save and load
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ModelArtifactManager(tmp_dir)

            vocabularies = {"user": user_vocab, "item": item_vocab}

            artifact = ModelArtifact(
                model_id="cat_only_model", model_name="Categorical Only", version=1
            )

            manager.save_model_artifact(
                model=model, artifact=artifact, vocabularies=vocabularies
            )

            # Predict with new data
            new_model = build_model_from_yaml(yaml_spec)
            state_dict, _ = manager.load_model_state_dict("cat_only_model")
            new_model.load_state_dict(state_dict)

            loaded_vocabs = manager.load_vocabularies("cat_only_model")

            new_users = ["user_2", "user_3"]
            new_items = ["item_a", "item_c"]

            new_user_idx = processor._execute_operator(
                "transform.label_encode",
                {"x": new_users, "state": loaded_vocabs["user"]},
            )["output"]

            new_item_idx = processor._execute_operator(
                "transform.label_encode",
                {"x": new_items, "state": loaded_vocabs["item"]},
            )["output"]

            new_model.eval()
            with torch.no_grad():
                scores = new_model(user=new_user_idx, item=new_item_idx)

            assert scores.shape == (2, 1)

    def test_oov_handling_in_prediction(self):
        """Test out-of-vocabulary handling during prediction."""
        # Train with limited vocabulary
        train_categories = ["cat_a", "cat_b", "cat_a", "cat_b"]
        features = torch.randn(4, 3)
        targets = torch.tensor([0.5, 1.5, 0.5, 1.5])

        processor = DataProcessor()
        fit_result = processor._execute_operator(
            "fit.label_encoder", {"x": train_categories}
        )
        vocab = fit_result["state"]

        train_idx = processor._execute_operator(
            "transform.label_encode", {"x": train_categories, "state": vocab}
        )["output"]

        # Simple model
        yaml_spec = """
        inputs:
          category:
            dtype: long
            shape: [null]
            categorical: true
            embedding_dim: 4
            vocab_size: 2

          features:
            dtype: float32
            shape: [null, 3]

        graph:
          - name: concat
            type: torch.cat
            params:
              dim: 1
            inputs: [category, features]

          - name: output
            type: torch.nn.Linear
            params:
              in_features: 7
              out_features: 1
            inputs:
              input: concat

        outputs:
          value: output.output
        """

        model = build_model_from_yaml(yaml_spec)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            preds = model(category=train_idx, features=features)
            loss = loss_fn(preds.squeeze(), targets)
            loss.backward()
            optimizer.step()

        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ModelArtifactManager(tmp_dir)
            artifact = ModelArtifact(
                model_id="oov_test", model_name="OOV Test", version=1
            )
            manager.save_model_artifact(
                model=model, artifact=artifact, vocabularies={"category": vocab}
            )

            # Load and predict with OOV value
            new_model = build_model_from_yaml(yaml_spec)
            state_dict, _ = manager.load_model_state_dict("oov_test")
            new_model.load_state_dict(state_dict)

            loaded_vocab = manager.load_vocabularies("oov_test")

            # Include an OOV category
            test_categories = ["cat_a", "cat_unknown", "cat_b"]
            test_features = torch.randn(3, 3)

            # Map OOV to index 0 (common pattern for "unknown/other" category)
            # This ensures the index is within the embedding layer's bounds
            test_idx = processor._execute_operator(
                "transform.label_encode",
                {
                    "x": test_categories,
                    "state": loaded_vocab["category"],
                    "unknown_value": 0,  # Map unknown to index 0
                },
            )["output"]

            # OOV value should be mapped to 0 as specified
            assert test_idx[1].item() == 0

            # Prediction should still work
            new_model.eval()
            with torch.no_grad():
                preds = new_model(category=test_idx, features=test_features)

            assert preds.shape == (3, 1)
