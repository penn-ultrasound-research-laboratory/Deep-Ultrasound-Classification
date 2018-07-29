class PatientSampleGeneratorException(Exception):
    """Raise for error instantiating PatientSampleGenerator due to malformatted input data"""

class ExtractSavePatientFeatureException(Exception):
    """Error extracting and saving patient feature to file"""

class TrainEvaluateLinearClassifierException(Exception):
    """Error training and evaluating linear classifier"""