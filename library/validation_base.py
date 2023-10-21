"""
Utilities for validation.

From arguments, we get some namespaces for validation arguments.

The external python file is required as --validation_executable argument.

We expect the python file to be executable and inherits ValidationBase class.

"""

from argparse import Namespace, ArgumentParser
from abc import ABCMeta, abstractmethod


class ValidationBase(metaclass=ABCMeta):
    """
    Base class for validation.
    
    ValidationBase class has the following methods:
    
        predict(self, args:Namespace) -> float:
            Return the score from args.
            We don't care how data is processed, but it has to get args as input, and return float.
            By implementation, you may skip using predict and implement log method directly.
            
        log(self, args:Namespace, log_dict:Dict):
            Log the score from args.
            It will append the score to log_dict then return it.
            
        start_validation(self):
            This method will be called at the start of validation.
            You can implement some initialization process here.
            This contains some loading cache, etc.
        
        end_validation(self):
            This method will be called at the end of validation.
            You can implement some finalization process here.
            This contains some cleaning up cache, etc.
            
        validate(self, args:Namespace, log_dict:Dict):
            Validate the model.
            This will be called from outside.
            
    """
    
    def __init__(self) -> None:
        pass
    
    def predict(self, args:Namespace) -> float:
        """
        Return the score from args.
        We don't care how data is processed, but it has to get args as input, and return float.
        By implementation, you may skip using predict and implement log method directly.
        """
        raise NotImplementedError("ValidationBase.predict is abstract method.")
    
    @abstractmethod
    def log(self, args:Namespace, log_dict:dict, **kwargs) -> dict:
        """
        Log the score from args.
        It will append the score to log_dict then return it.
        """
        score = self.predict(args)
        log_dict["val/score"] = score
        return log_dict
    
    def start_validation(self, args:Namespace, **kwargs):
        """
        This method will be called at the start of validation.
        You can implement some initialization process here.
        This contains some loading cache, etc.
        """
        pass   
    
    def end_validation(self, args:Namespace, **kwargs):
        """
        This method will be called at the end of validation.
        You can implement some finalization process here.
        This contains some cleaning up cache, etc.
        """
        pass
    
    def validate(self, args:Namespace, log_dict:dict, **kwargs) -> dict:
        """
        Validate the model.
        """
        self.start_validation(args, **kwargs)
        retval = self.log(args, log_dict, **kwargs)
        self.end_validation(args, **kwargs)
        return retval
    
class ValidatorArgumentParserCallback:
    """
    Callback class for argparse.ArgumentParser.
    """
    
    def add_callback(parser:ArgumentParser):
        """
        Adds arguments for validation.
        ex:
            parser.add_argument("--validation_executable", type=str, default="library/validation_clip.py", help="Path to validation executable.")
        """
        pass