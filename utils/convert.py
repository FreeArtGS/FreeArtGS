import numpy as np

def convert_dict_to_json_format(obj):
        if isinstance(obj, dict):
            return {k: convert_dict_to_json_format(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_dict_to_json_format(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        else:
            return obj