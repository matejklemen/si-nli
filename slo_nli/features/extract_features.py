from typing import Optional, List

from transformers import FeatureExtractionPipeline


class AugmentedFeatureExtractionPipeline(FeatureExtractionPipeline):
	def __init__(self, model: "PreTrainedModel", use_layers: Optional[List[int]] = None,
				 **kwargs):
		self.return_layers = [-1] if use_layers is None else use_layers
		assert isinstance(self.return_layers, list)

		# Using a custom postprocessing script
		assert model.config.output_hidden_states, "output_hidden_states=True required in the used model"

		kwargs["model"] = model
		super().__init__(**kwargs)
		assert self.framework == "pt", "Modified code not implemented for TensorFlow"

	def postprocess(self, model_outputs):
		# List of size: [num_examples, num_layers, num_tokens, hidden_size], i.e. list of lists
		return [model_outputs.hidden_states[_layer][0].tolist() for _layer in self.return_layers]
