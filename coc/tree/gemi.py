import os
import asyncio
import logging
import base64
import io
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Iterator, Union
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from coc.data.fulltask import FullTask
from coc.tool.task import Task
from coc.tree.une import TreeNode, root_factory, CodeList
from coc.util.text import extract_code, extract_boxed
from coc.config import MAX_DEPTH, BRANCH_WIDTH, TEMPERATURE

# Import from coc.tool.vqa for Gemini functionality
from coc.tool.vqa.gemini import Gemini
from coc.tool.vqa import gemini_as_llm

# Import Exec for code execution
from coc.exec.mod import Exec

# Import 3D vision modules if available - these will be optional
try:
    # Try direct import first (using the specific paths in this codebase)
    from coc.tool.depth import DepthAnything
except ImportError:
    try:
        # Try importing from submodules if not in coc.tool
        from depth_anything_v2.dpt import DepthAnything
    except ImportError:
        logging.warning("DepthAnything module could not be imported")
        DepthAnything = None

try:
    from coc.tool._3d import Fast3R
except ImportError:
    try:
        from fast3r.fast3r import Fast3R
    except ImportError:
        logging.warning("Fast3R module could not be imported")
        Fast3R = None

try:
    from coc.tool._3d import PoinTr
except ImportError:
    try:
        from pointr.models.pointr import PoinTr
    except ImportError:
        logging.warning("PoinTr module could not be imported")
        PoinTr = None

try:
    from coc.tool.sam import sam_model_registry
except ImportError:
    try:
        from sam2.build_sam import sam_model_registry
    except ImportError:
        logging.warning("SAM2 module could not be imported")
        sam_model_registry = None

# Set up logging
logger = logging.getLogger(__name__)

class DepthEstimator:
    """Wrapper for Depth-Anything-V2 depth estimation model"""
    
    def __init__(self, model_type: str = "large"):
        """Initialize depth estimation model
        
        Args:
            model_type: Size of model to use ('small', 'base', 'large')
        """
        if DepthAnything is None:
            raise ImportError("DepthAnything module is not available")
            
        self.model = DepthAnything.from_pretrained(f"depth-anything-v2-{model_type}")
        self.model.eval()
    
    def predict(self, image: Image.Image) -> Image.Image:
        """Predict depth map for a single image
        
        Args:
            image: PIL Image to predict depth for
            
        Returns:
            Depth map as PIL Image
        """
        # Convert to model input format and predict
        depth_map = self.model.infer_image(image)
        
        # Convert to PIL Image
        depth_image = Image.fromarray(depth_map)
        return depth_image

class SegmentationModel:
    """Wrapper for SAM2 segmentation model"""
    
    def __init__(self, model_type: str = "vit_h"):
        """Initialize segmentation model
        
        Args:
            model_type: Type of SAM2 model to use
        """
        if sam_model_registry is None:
            raise ImportError("SAM2 module is not available")
            
        self.model = sam_model_registry[model_type]()
        self.model.eval()
    
    def segment(self, image: Image.Image, points: List[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """Segment objects in an image
        
        Args:
            image: PIL Image to segment
            points: Optional list of points to use for prompting
            
        Returns:
            List of segmentation masks
        """
        # Convert inputs to model format and get predictions
        masks = self.model.predict(image, point_coords=points)
        return masks

class Novel3DViewSynthesis:
    """Wrapper for Fast3R novel view synthesis model"""
    
    def __init__(self):
        """Initialize novel view synthesis model"""
        if Fast3R is None:
            raise ImportError("Fast3R module is not available")
            
        self.model = Fast3R()
        self.model.eval()
    
    def synthesize_novel_views(self, image: Image.Image, depth_map: Image.Image, 
                              angles: List[float] = None) -> List[Image.Image]:
        """Generate novel views of a scene
        
        Args:
            image: Source RGB image
            depth_map: Corresponding depth map
            angles: List of camera angles to render (in degrees)
            
        Returns:
            List of novel view images
        """
        if angles is None:
            # Default angles: 15 degrees left, right, up, down
            angles = [-15, 15, -15, 15]  
        
        novel_views = self.model.render_novel_views(image, depth_map, angles)
        return novel_views

class PointCloudProcessor:
    """Wrapper for PoinTr point cloud processing"""
    
    def __init__(self):
        """Initialize point cloud processor"""
        if PoinTr is None:
            raise ImportError("PoinTr module is not available")
            
        self.model = PoinTr()
        self.model.eval()
    
    def generate_point_cloud(self, image: Image.Image, depth_map: Image.Image) -> Dict[str, Any]:
        """Generate 3D point cloud from image and depth map
        
        Args:
            image: Source RGB image
            depth_map: Corresponding depth map
            
        Returns:
            Point cloud data
        """
        point_cloud = self.model.image_to_pointcloud(image, depth_map)
        return point_cloud

class GeminiAgent:
    """
    A multimodal agent that uses Gemini with enhanced 3D vision capabilities
    and sophisticated vision tooling
    """
    
    def __init__(self, 
                 use_depth: bool = True,
                 use_segmentation: bool = True,
                 use_novel_view: bool = True,
                 use_point_cloud: bool = True,
                 verbose: bool = False):
        """
        Initialize the Gemini agent with 3D vision capabilities
        
        Args:
            use_depth: Whether to use depth estimation
            use_segmentation: Whether to use segmentation
            use_novel_view: Whether to use novel view synthesis
            use_point_cloud: Whether to use point cloud processing
            verbose: Whether to print detailed logs
        """
        # Initialize Gemini VQA tool
        self.gemini = Gemini()
        self.verbose = verbose
        
        # Check which modules are available
        self.modules_available = {
            "depth": DepthAnything is not None,
            "segmentation": sam_model_registry is not None,
            "novel_view": Fast3R is not None,
            "point_cloud": PoinTr is not None
        }
        
        # Initialize vision modules as needed (with graceful degradation)
        self.use_depth = use_depth and self.modules_available["depth"]
        self.use_segmentation = use_segmentation and self.modules_available["segmentation"]
        self.use_novel_view = use_novel_view and self.modules_available["novel_view"]
        self.use_point_cloud = use_point_cloud and self.modules_available["point_cloud"]
        
        # Log available capabilities
        capabilities = []
        if self.use_depth:
            capabilities.append("depth estimation")
            try:
                self.depth_estimator = DepthEstimator()
            except Exception as e:
                logging.error(f"Failed to initialize depth estimator: {e}")
                self.use_depth = False
        
        if self.use_segmentation:
            capabilities.append("segmentation")
            try:
                self.segmentation_model = SegmentationModel()
            except Exception as e:
                logging.error(f"Failed to initialize segmentation model: {e}")
                self.use_segmentation = False
        
        if self.use_novel_view:
            capabilities.append("novel view synthesis")
            try:
                self.novel_view_model = Novel3DViewSynthesis()
            except Exception as e:
                logging.error(f"Failed to initialize novel view synthesis: {e}")
                self.use_novel_view = False
            
        if self.use_point_cloud:
            capabilities.append("point cloud processing")
            try:
                self.point_cloud_processor = PointCloudProcessor()
            except Exception as e:
                logging.error(f"Failed to initialize point cloud processor: {e}")
                self.use_point_cloud = False
        
        if capabilities:
            logging.info(f"GeminiAgent initialized with capabilities: {', '.join(capabilities)}")
        else:
            logging.warning("GeminiAgent initialized without any 3D vision capabilities")
        
        # Initialize counters for tool calls
        self.tool_call_counts = {
            "segmentation": 0,
            "depth": 0,
            "novel_view": 0,
            "point_cloud": 0,
            "zoomed": 0,
            "cropped": 0,
            "counting": 0
        }

    def _get_prompt_capabilities(self):
        """Get description of active capabilities for system prompt"""
        capabilities = []
        if self.use_depth:
            capabilities.append("depth estimation")
        if self.use_segmentation:
            capabilities.append("segmentation")
        if self.use_novel_view:
            capabilities.append("novel view synthesis")
        if self.use_point_cloud:
            capabilities.append("3D point cloud data")
        
        if not capabilities:
            return "You are a helpful multimodal assistant."
        
        # Enhanced prompt with function interfaces for vision tools
        function_interfaces = '''
```python
from typing import *
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Core vision analysis functions
def segment_image(image: Image.Image, prompt: Optional[str] = None) -> List[np.ndarray]:
    """Segment objects in the image using SAM model
    
    Args:
        image: Input image
        prompt: Optional text prompt to guide segmentation
        
    Returns:
        List of segmentation masks as boolean arrays
    """
    ...

def color_based_segmentation(image: Image.Image, color_ranges: Optional[List] = None) -> List[np.ndarray]:
    """Segment image based on color ranges
    
    Args:
        image: Input image
        color_ranges: List of (lower, upper) HSV color ranges
        
    Returns:
        List of binary masks for each color range
    """
    ...

def crop_and_zoom(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    """Crop and zoom into a region of interest
    
    Args:
        image: Input image
        bbox: Bounding box as (left, top, right, bottom) in normalized coordinates
        
    Returns:
        Cropped image
    """
    ...

# Advanced vision functions
def get_depth_map(image: Image.Image) -> Image.Image:
    """Generate depth map from image
    
    Args:
        image: Input image
        
    Returns:
        Depth map as grayscale image
    """
    ...

def generate_novel_view(image: Image.Image, depth_map: Image.Image, angle: float) -> Image.Image:
    """Generate novel view of scene from different angle
    
    Args:
        image: Input RGB image
        depth_map: Corresponding depth map
        angle: Camera angle in degrees
        
    Returns:
        Novel view image
    """
    ...

def generate_point_cloud(image: Image.Image, depth_map: Image.Image) -> Dict[str, Any]:
    """Generate 3D point cloud from image and depth map
    
    Args:
        image: Input RGB image
        depth_map: Corresponding depth map
        
    Returns:
        Point cloud data dictionary
    """
    ...

# Utility functions
def visualize_results(image: Image.Image, masks: List[np.ndarray], counts: Dict[str, int]) -> Image.Image:
    """Visualize analysis results on the image
    
    Args:
        image: Original image
        masks: Segmentation masks
        counts: Object counts by category
        
    Returns:
        Visualization image
    """
    ...

def cross_validate_counts(counts_by_method: Dict[str, int]) -> Tuple[int, float]:
    """Cross-validate counts from different methods
    
    Args:
        counts_by_method: Dictionary of counts by method name
        
    Returns:
        Tuple of (best_count, confidence)
    """
    ...
```
'''
        
        return (
            "You are a powerful multimodal AI with advanced 3D vision capabilities. "
            f"You have access to specialized vision tools including: {', '.join(capabilities)}. "
            "\n\nIMPORTANT: You currently do not have direct visual understanding capabilities. "
            "You MUST solve vision tasks by EXPLICITLY USING THESE TOOLS THROUGH CODE GENERATION. "
            "Simply describing what you 'see' in images is NOT an option - you must analyze them through code."
            "\n\nYou may access visual tools by writing Python code. Your task will be presented to you, "
            "and you must write Python code snippets to interact with these tools. The code will be "
            "executed, and the output will be provided back to you to guide your reasoning."
            f"\n\nAvailable vision tool interfaces (implement your analysis using these):\n{function_interfaces}"
            "\n\nFor vision tasks, particularly counting objects:"
            "\n1. Break the task into smaller steps that can be solved with code"
            "\n2. Use segmentation with contour analysis to isolate individual objects"
            "\n3. Apply multiple techniques and cross-verify your results (e.g., edge detection + color segmentation)"
            "\n4. Employ cropping and zooming to focus on regions with ambiguous counts"
            
            "\n\nWhen facing intricate problems such as mottled/irregular patterns or detailed textures:"
            "\n- Try multiple threshold values and merge results"
            "\n- Perform region-by-region analysis with zooming on complex areas"
            
            "\n\nYour code must be executable Python and thoroughly analyze the visual data. "
            "You MUST write Python code to complete the task - DO NOT try to answer directly without using code."
            "\n\nTo use the code correctly:"
            "\n1. Write self-contained Python code snippets"
            "\n2. Use 'image' or 'image_0', 'image_1', etc. to refer to the provided images"
            "\n3. Print your results to view them"
            "\n4. Always enclose your code in triple backticks with Python specification"
        )
        
    # Method to crop and zoom into a region of interest
    def crop_and_zoom(self, image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
        """
        Crop and zoom into a region of interest in the image
        
        Args:
            image: Original image
            bbox: Bounding box in format (left, top, right, bottom) - normalized 0-1 coordinates
            
        Returns:
            Cropped and zoomed image
        """
        self.tool_call_counts["cropped"] += 1
        
        # Convert normalized coordinates to pixel values
        width, height = image.size
        left = int(bbox[0] * width)
        top = int(bbox[1] * height)
        right = int(bbox[2] * width)
        bottom = int(bbox[3] * height)
        
        # Ensure coordinates are valid
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
        
        # Crop the image
        cropped = image.crop((left, top, right, bottom))
        
        if self.verbose:
            logging.info(f"Cropped image to region {bbox}, new size: {cropped.size}")
            
        return cropped
    
    # Method to segment and count objects of a specific type
    def segment_and_count_objects(self, image: Image.Image, object_prompt: str) -> Tuple[int, Image.Image]:
        """
        Segment and count objects of a specific type in the image
        
        Args:
            image: Image to analyze
            object_prompt: Description of the objects to count
            
        Returns:
            Tuple of (count, segmentation visualization)
        """
        self.tool_call_counts["counting"] += 1
        self.tool_call_counts["segmentation"] += 1
        
        if not self.use_segmentation:
            # If segmentation not available, use Gemini as fallback
            text_response = self.gemini.run_freestyle([
                f"Count the number of {object_prompt} in this image. Return ONLY the number:",
                image
            ])
            
            try:
                count = int(text_response.strip())
            except:
                # If not a clean number, use a conservative estimate
                count = 0
            
            if self.verbose:
                logging.info(f"Counted {count} {object_prompt} using Gemini fallback")
                
            return count, image
        
        try:
            # Use segmentation to identify objects
            masks = self.segmentation_model.segment(image)
            
            # Get Gemini to identify which segments match the object type
            # Convert masks to visualizations
            mask_vis = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            
            for i, mask in enumerate(masks):
                # Apply a unique color to each segment
                color = [(i*50) % 255, ((i+1)*50) % 255, ((i+2)*50) % 255]
                for c in range(3):
                    mask_vis[:,:,c] = np.where(mask, color[c], mask_vis[:,:,c])
            
            # Convert to PIL Image
            mask_image = Image.fromarray(mask_vis)
            
            # Ask Gemini to identify which segments match the object type
            response = self.gemini.run_freestyle([
                f"This is a segmentation map of an image. Identify which segments (colored regions) correspond to {object_prompt}. Count them and return ONLY the number of distinct {object_prompt}:",
                mask_image,
                image
            ])
            
            try:
                count = int(response.strip())
            except:
                # Use a heuristic estimation if Gemini's response isn't a clean number
                count = len(masks) // 2  # Conservative estimate
            
            if self.verbose:
                logging.info(f"Counted {count} {object_prompt} using segmentation")
                
            return count, mask_image
            
        except Exception as e:
            logging.error(f"Error in segment_and_count_objects: {e}")
            # Fallback to direct Gemini counting
            text_response = self.gemini.run_freestyle([
                f"Count the number of {object_prompt} in this image. Return ONLY the number:",
                image
            ])
            
            try:
                count = int(text_response.strip())
            except:
                count = 0
                
            return count, image

    def _generate_enhanced_vision_data(self, images: List[Image.Image]) -> Dict[str, Any]:
        """
        Generate enhanced vision data from images
        
        Args:
            images: List of PIL Images
            
        Returns:
            Dictionary of enhanced vision data
        """
        enhanced_data = {}
        
        # Check if images are provided
        if not images:
            return enhanced_data
        
        # Check if we have any vision modules available
        if not any([self.use_depth, self.use_segmentation, self.use_novel_view, self.use_point_cloud]):
            logger.warning("No vision enhancement modules available - returning original images only")
            return enhanced_data
        
        try:
            # Generate depth maps
            if self.use_depth:
                try:
                    depth_maps = []
                    for image in images:
                        depth_map = self.depth_estimator.predict(image)
                        depth_maps.append(depth_map)
                    enhanced_data["depth_maps"] = depth_maps
                    self.tool_call_counts["depth"] += 1
                except Exception as e:
                    logger.error(f"Failed to generate depth maps: {e}")
            
            # Generate segmentation
            if self.use_segmentation:
                try:
                    segmentation_results = []
                    segmentation_summary = []
                    
                    for image in images:
                        masks = self.segmentation_model.segment(image)
                        segmentation_results.append(masks)
                        
                        # Create textual summary of segmentation
                        summary = f"Detected {len(masks)} objects/regions in the image."
                        segmentation_summary.append(summary)
                    
                    enhanced_data["segmentation_results"] = segmentation_results
                    enhanced_data["segmentation_summary"] = "; ".join(segmentation_summary)
                    self.tool_call_counts["segmentation"] += 1
                except Exception as e:
                    logger.error(f"Failed to generate segmentation: {e}")
            
            # Generate novel views
            if self.use_novel_view and self.use_depth and "depth_maps" in enhanced_data:
                try:
                    novel_views = []
                    for i, image in enumerate(images):
                        if i < len(enhanced_data.get("depth_maps", [])):
                            depth_map = enhanced_data["depth_maps"][i]
                            views = self.novel_view_model.synthesize_novel_views(image, depth_map)
                            novel_views.extend(views)
                    
                    enhanced_data["novel_views"] = novel_views
                    self.tool_call_counts["novel_view"] += 1
                except Exception as e:
                    logger.error(f"Failed to generate novel views: {e}")
            
            # Generate point cloud data
            if self.use_point_cloud and self.use_depth and "depth_maps" in enhanced_data:
                try:
                    point_cloud_data = []
                    point_cloud_summary = []
                    
                    for i, image in enumerate(images):
                        if i < len(enhanced_data.get("depth_maps", [])):
                            depth_map = enhanced_data["depth_maps"][i]
                            point_cloud = self.point_cloud_processor.generate_point_cloud(image, depth_map)
                            point_cloud_data.append(point_cloud)
                            
                            # Create textual summary of point cloud
                            summary = f"3D point cloud generated with {point_cloud.get('num_points', 0)} points."
                            point_cloud_summary.append(summary)
                    
                    enhanced_data["point_cloud_data"] = point_cloud_data
                    enhanced_data["point_cloud_summary"] = "; ".join(point_cloud_summary)
                    self.tool_call_counts["point_cloud"] += 1
                except Exception as e:
                    logger.error(f"Failed to generate point cloud data: {e}")
        
        except Exception as e:
            logger.error(f"Error generating enhanced vision data: {e}")
        
        return enhanced_data

    def generate(self, prompt: str, images: List[Image.Image] = None) -> str:
        """
        Generate a response using Gemini with enhanced 3D vision capabilities
        
        Args:
            prompt: Text prompt
            images: List of PIL Images
            
        Returns:
            Text response from the model
        """
        # For complex counting tasks, use code execution approach
        counting_indicators = ["count", "how many", "number of", "total of", "tally", "enumerate"]
        if images and any(indicator in prompt.lower() for indicator in counting_indicators):
            return self.analyze_with_code_execution(prompt, images)
            
        # Generate enhanced vision data
        enhanced_data = {}
        if images:
            try:
                enhanced_data = self._generate_enhanced_vision_data(images)
            except Exception as e:
                logging.warning(f"Enhanced vision data generation failed: {e}")
        
        # Build the inputs list for freestyle
        inputs = []
        
        # Add the prompt with capability information
        full_prompt = prompt
        
        # Add enhanced data descriptions to the prompt if available
        if enhanced_data and any(key in enhanced_data for key in ["segmentation_summary", "point_cloud_summary"]):
            capability_info = "\n\nAdditional 3D information:\n"
            
            if "segmentation_summary" in enhanced_data:
                capability_info += f"- Segmentation: {enhanced_data['segmentation_summary']}\n"
            
            if "point_cloud_summary" in enhanced_data:
                capability_info += f"- 3D Structure: {enhanced_data['point_cloud_summary']}\n"
            
            # Add the system prompt
            full_prompt = f"{self._get_prompt_capabilities()}\n\n{prompt}{capability_info}"
        
        inputs.append(full_prompt)
        
        # Add the original images
        if images:
            inputs.extend(images)
        
        # Add enhanced vision images
        if enhanced_data:
            # Add depth maps
            if "depth_maps" in enhanced_data:
                for i, depth_map in enumerate(enhanced_data["depth_maps"]):
                    inputs.append("Depth map showing distance of objects from the camera:")
                    inputs.append(depth_map)
            
            # Add novel views
            if "novel_views" in enhanced_data:
                for i, novel_view in enumerate(enhanced_data["novel_views"]):
                    inputs.append(f"Alternative viewpoint {i+1} of the scene:")
                    inputs.append(novel_view)
        
        # Call Gemini to process the request
        try:
            response = self.gemini.run_freestyle(inputs)
            return response
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def run_freestyle(self, inputs: List[Union[str, Image.Image]]) -> str:
        """
        Run the model with a mix of text and images
        
        Args:
            inputs: List of strings and PIL Images
            
        Returns:
            Model response
        """
        if not inputs:
            return ""
        
        # Use the Gemini run_freestyle method directly
        try:
            return self.gemini.run_freestyle(inputs)
        except Exception as e:
            logging.error(f"Error in run_freestyle: {e}")
            return f"Error: {str(e)}"

    def generate_branches(self, prompts: List[str], images: List[Image.Image] = None) -> List[str]:
        """
        Generate multiple responses for different prompts
        
        Args:
            prompts: List of text prompts
            images: List of images to include with each prompt
            
        Returns:
            List of model responses
        """
        responses = []
        
        for prompt in prompts:
            response = self.generate(prompt, images)
            responses.append(response)
        
        return responses

    def generate_orchestrated(self, prompt: str, images: List[Image.Image] = None) -> str:
        """
        Generate a response using a multi-step orchestrated approach with tool calls
        
        Args:
            prompt: Text prompt
            images: List of PIL Images
            
        Returns:
            Text response from the model
        """
        if not images:
            # Text-only query, use standard generate
            return self.generate(prompt, images)
        
        # Reset tool call counters
        for key in self.tool_call_counts:
            self.tool_call_counts[key] = 0
        
        # Use code execution for complex vision tasks (counting, retail analysis)
        # Check if this is a counting task
        counting_indicators = [
            "count", "how many", "number of", 
            "total of", "tally", "enumerate"
        ]
        
        is_counting_task = any(indicator in prompt.lower() for indicator in counting_indicators)
        
        # For complex counting/retail tasks, use code execution approach
        if is_counting_task:
            return self.analyze_with_code_execution(prompt, images)
        
        # For other tasks, use the original orchestrated approach
        # Extract image for analysis (use first image if multiple)
        image = images[0]
        
        enhanced_data = {}
        enhanced_results = []
        if is_counting_task:
            # Generic counting task - figure out what to count
            object_to_count = self.gemini.run_freestyle([
                f"Based on this query: '{prompt}', what specific objects should I count in the image? Answer with just the object type/category:",
                image
            ])
            
            # Count the objects
            count, seg_vis = self.segment_and_count_objects(image, object_to_count)
            
            enhanced_results.append(f"COUNTING ANALYSIS:")
            enhanced_results.append(f"Object type: {object_to_count}")
            enhanced_results.append(f"Count: {count}")
        
        elif self.use_depth or self.use_segmentation:
            # For other tasks that might benefit from vision enhancements
            try:
                enhanced_data = self._generate_enhanced_vision_data(images)
            except Exception as e:
                logging.warning(f"Enhanced vision data generation failed: {e}")

        # Prepare the enhanced context
        enhanced_prompt = prompt
        if enhanced_results:
            enhanced_context = "\n".join(enhanced_results)
            enhanced_prompt = f"I've analyzed the image with specialized vision tools. Here are the results:\n\n{enhanced_context}\n\nBased on this analysis, please answer the original question: {prompt}"

        # Call Gemini with all the enhanced data
        inputs = [enhanced_prompt]
        
        # Add the original images
        if images:
            inputs.extend(images)
        
        # Add segmentation visualizations if available
        if is_counting_task or is_retail_task:
            if 'seg_vis' in locals() and seg_vis is not None:
                inputs.append("Segmentation visualization of counted objects:")
                inputs.append(seg_vis)
        
        # Add enhanced vision data if available
        if enhanced_data:
            # Add depth maps
            if "depth_maps" in enhanced_data:
                for i, depth_map in enumerate(enhanced_data["depth_maps"]):
                    inputs.append("Depth map showing distance of objects from the camera:")
                    inputs.append(depth_map)
            
            # Add novel views
            if "novel_views" in enhanced_data:
                for i, novel_view in enumerate(enhanced_data["novel_views"]):
                    inputs.append(f"Alternative viewpoint {i+1} of the scene:")
                    inputs.append(novel_view)
        
        # Generate the final response
        try:
            if self.verbose:
                logging.info(f"Tool call counts: {self.tool_call_counts}")
            
            response = self.gemini.run_freestyle(inputs)
            return response
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    # New method to set up Exec environment for code generation
    def setup_exec_environment(self):
        """Set up an execution environment for code generation
        
        Returns:
            Exec environment with vision tools
        """
        from coc.exec.mod import Exec
        
        # Create Exec instance
        exec_env = Exec()
        
        # Add safe color conversion utility to avoid OpenCV errors
        safe_conversion_code = """
        def safe_convert_to_gray(image):
            '''Safely convert an image to grayscale, handling already-grayscale images'''
            import cv2
            import numpy as np
            
            # Check image type
            if isinstance(image, np.ndarray):
                # Check if image is already grayscale (2D array)
                if len(image.shape) == 2:
                    return image
                # Convert BGR/RGB to grayscale if 3 channels
                elif len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    raise ValueError(f"Unexpected image shape: {image.shape}")
            else:
                # Try to convert to numpy array if not already
                try:
                    import numpy as np
                    from PIL import Image
                    if isinstance(image, Image.Image):
                        # Convert PIL Image to numpy array
                        np_image = np.array(image)
                        # Handle grayscale vs RGB
                        if len(np_image.shape) == 2:
                            return np_image
                        else:
                            return cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                except Exception as e:
                    print(f"Error in safe_convert_to_gray: {e}")
                    # Return empty image as fallback
                    return np.zeros((10, 10), dtype=np.uint8)
        """
        
        # Basic setup code for vision tasks
        setup_code = """
        import numpy as np
        import cv2
        from PIL import Image
        import matplotlib.pyplot as plt
        import io
        
        # Utility functions for image processing
        def visualize_results(image, regions=None, labels=None, title=None):
            '''Visualize image with optional regions and labels'''
            plt.figure(figsize=(10, 10))
            
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2 or image.shape[2] == 1:
                    plt.imshow(image, cmap='gray')
                else:
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(image)
                
            if regions is not None:
                for i, region in enumerate(regions):
                    y, x, h, w = region
                    rect = plt.Rectangle((x, y), w, h, 
                                         fill=False, 
                                         edgecolor='red', 
                                         linewidth=2)
                    plt.gca().add_patch(rect)
                    
                    if labels is not None and i < len(labels):
                        plt.text(x, y-5, labels[i], 
                                 color='red', 
                                 fontsize=12, 
                                 bbox=dict(facecolor='white', alpha=0.7))
            
            if title:
                plt.title(title)
                
            plt.axis('off')
            
            # Save plot to a buffer and return as PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return Image.open(buf)
        
        def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
            '''Overlay a binary mask on an image'''
            # Convert PIL Image to cv2 if needed
            if isinstance(image, Image.Image):
                image_cv = np.array(image)
                if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            else:
                image_cv = image.copy()
            
            # Create RGB overlay
            overlay = image_cv.copy()
            
            # Apply color to mask regions
            if len(image_cv.shape) == 2:
                # Convert grayscale to BGR for overlay
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
            
            if mask.dtype != bool:
                mask = mask > 0
                
            overlay[mask] = color
            
            # Blend images
            output = cv2.addWeighted(overlay, alpha, image_cv, 1-alpha, 0)
            return output
        """
        
        # Execute the setup code
        stdout, stderr, _ = exec_env._run(setup_code)
        if stderr:
            logging.warning(f"Setup code error: {stderr}")
        
        # Execute the safe conversion utility
        stdout, stderr, _ = exec_env._run(safe_conversion_code)
        if stderr:
            logging.warning(f"Safe conversion code error: {stderr}")
        
        # Add the segmentation model if available
        if self.use_segmentation and self.modules_available["segmentation"]:
            try:
                exec_env.set_var("segmentation_model", self.segmentation_model)
                # Add helper function to run segmentation
                segmentation_code = """
                def segment_image(image, prompt=None):
                    '''Segment the image using SAM model'''
                    return segmentation_model.segment(image, points=None)
                """
                exec_env._run(segmentation_code)
            except Exception as e:
                logging.error(f"Failed to add segmentation to Exec: {e}")
        
        # Add the depth estimator if available
        if self.use_depth and self.modules_available["depth"]:
            try:
                exec_env.set_var("depth_estimator", self.depth_estimator)
                # Add helper function to get depth map
                depth_code = """
                def get_depth_map(image):
                    '''Get depth map for an image'''
                    return depth_estimator.predict(image)
                """
                exec_env._run(depth_code)
            except Exception as e:
                logging.error(f"Failed to add depth estimation to Exec: {e}")
        
        return exec_env
    
    # New method to use code generation for vision tasks
    def analyze_with_code_execution(self, prompt: str, images: List[Image.Image]) -> str:
        """Analyze images using code generation and execution
        
        Args:
            prompt: Text prompt describing the task
            images: List of images to analyze
            
        Returns:
            Analysis result
        """
        if not images:
            return self.generate(prompt, images)
        
        # Set up the Exec environment
        exec_env = self.setup_exec_environment()
        
        # Add the images to the environment - IMPORTANT: save actual image files to disk
        # to avoid the "Image file not found" error when the generated code attempts to load them
        for i, img in enumerate(images):
            # Save the image directly as a variable in the environment
            exec_env.set_var(f"image_{i}", img)
            
            # Also save an actual file on disk in case the code attempts to reload it
            try:
                temp_dir = os.path.join(os.getcwd(), "data", "temp_images")
                os.makedirs(temp_dir, exist_ok=True)
                img_path = os.path.join(temp_dir, f"image_{i}.jpg")
                img.save(img_path)
                exec_env.set_var(f"image_{i}_path", img_path)
            except Exception as e:
                logging.warning(f"Failed to save image_{i} to disk: {e}")
        
        # Create a list of all image variables
        exec_env.set_var("images", [exec_env.get_var(f"image_{i}") for i in range(len(images))])
        
        # Get function interfaces for the prompt
        full_prompt = self._get_prompt_capabilities()
        try:
            # Try to extract the function interfaces section if it exists
            if "Available vision tool interfaces" in full_prompt:
                function_interfaces_text = full_prompt.split("Available vision tool interfaces")[1].split("For vision tasks")[0].strip()
            else:
                # Otherwise extract function interfaces directly from the function_interfaces attribute
                function_interfaces_text = '''
```python
from typing import *
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Core vision analysis functions
def segment_image(image: Image.Image, prompt: Optional[str] = None) -> List[np.ndarray]:
    """Segment objects in the image using SAM model"""
    ...

def color_based_segmentation(image: Image.Image, color_ranges: Optional[List] = None) -> List[np.ndarray]:
    """Segment image based on color ranges"""
    ...

def crop_and_zoom(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    """Crop and zoom into a region of interest"""
    ...
```
'''
        except Exception as e:
            # In case of any error, use a simplified interface
            logging.warning(f"Error extracting function interfaces: {e}")
            function_interfaces_text = '''
```python
# Basic image processing functions
def segment_image(image): ...
def color_based_segmentation(image): ...
def crop_and_zoom(image, bbox): ...
```
'''
        
        # Base code generation prompt template
        code_gen_template = f"""
        Write Python code to analyze the given image(s) for this task:
        
        "{prompt}"
        
        You should utilize the available vision tools through the following interfaces:
        
        {function_interfaces_text}
        
        Your code should:
        1. Use appropriate vision analysis functions from the interface
        2. Process the image systematically to extract relevant information
        3. Implement a step-by-step approach with clear intermediate results
        4. Handle potential errors and edge cases
        5. Print final results and any important metrics
        
        The images are available as 'image_0', 'image_1', etc. and also as a list 'images'.
        
        IMPORTANT: Your code should be executable Python. Do not include explanations or comments -
        focus only on implementing a robust solution that works.
        """
        
        code_generation_prompt = code_gen_template
        
        # Get code from Gemini
        generated_code = self.gemini.run_freestyle(code_generation_prompt)
        
        # Extract just the code
        code = extract_code(generated_code) or extract_boxed(generated_code) or generated_code
        
        # Execute the generated code
        try:
            stdout, stderr, displayed_images = exec_env._run(code)
            
            # If there are errors, try to debug and fix the code
            if stderr and "Error" in stderr:
                debug_prompt = f"""
                The following code produced errors:
                
                ```python
                {code}
                ```
                
                Error:
                {stderr}
                
                Please fix the code to address these errors. Make sure to use the available functions properly.
                The vision tool interfaces are:
                
                {function_interfaces_text}
                
                ONLY provide the corrected code with no explanations.
                """
                
                fixed_code = self.gemini.run_freestyle(debug_prompt)
                fixed_code = extract_code(fixed_code) or extract_boxed(fixed_code) or fixed_code
                
                # Try running the fixed code
                stdout, stderr, displayed_images = exec_env._run(fixed_code)
                
                # Update the code reference for later
                if not stderr or "Error" not in stderr:
                    code = fixed_code
            
            # Prepare the result
            result = ""
            if stdout:
                result += f"{stdout}\n\n"
            
            if stderr:
                result += f"Error in analysis: {stderr}\n\n"
            
            # Ask Gemini to interpret the results
            interpretation_prompt = f"""
            I ran the image analysis with this code:
            
            ```python
            {code}
            ```
            
            And got these results:
            
            {stdout if stdout else 'No standard output'}
            
            Based on these results, answer the original question:
            
            "{prompt}"
            
            Provide a concise, accurate answer using only the data from the analysis.
            Be extremely precise with counts and numbers - do not round or approximate.
            If the analysis identifies different counts from different methods, explain
            which is most reliable and why.
            """
            
            interpretation = self.gemini.run_freestyle(interpretation_prompt)
            result += interpretation
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing generated code: {e}")
            return f"Error analyzing images with code execution: {str(e)}"

def generate_one_child(parent: TreeNode, suggestive_hint: str, agent: GeminiAgent) -> tuple[TreeNode, Optional[str]]:
    """
    Generate a single child node using the Gemini agent
    
    Args:
        parent: Parent node
        suggestive_hint: Hint for generation
        agent: GeminiAgent instance
        
    Returns:
        New child node and error message if any
    """
    # Construct prompt
    prompt = (
        f"Given the following code:\n```python\n{parent.curr_code if hasattr(parent, 'curr_code') else (parent._[0].code if parent._ and len(parent._) > 0 else '')}\n```\n"
        f"Your task: {suggestive_hint}\n"
        "Generate improved code that addresses this task. "
        "The code should be complete, well-structured, and functional. "
        "Focus on explicitly using vision tools through code generation rather than relying on "
        "simple model observations. For counting tasks, use segmentation and contour analysis. "
        "For retail analysis, divide images into layers for accurate counting."
    )
    
    # Get response
    try:
        # Prepare images if available
        images = None
        if hasattr(parent, 'images') and parent.images:
            images = parent.images
        
        # Use the analyze_with_code_execution for better vision-based tasks
        if images and ("count" in suggestive_hint.lower() or "retail" in suggestive_hint.lower()):
            response = agent.analyze_with_code_execution(prompt, images)
        else:
            response = agent.generate(prompt, images)
            
        code = extract_code(response) or extract_boxed(response) or response
        
        # Create child node
        child = parent.create_child(code)
        return child, None
    
    except Exception as e:
        error_msg = f"Error generating child: {str(e)}"
        logger.error(error_msg)
        return parent, error_msg

def generate_children(nodes_with_code: list[TreeNode], num_children: int, agent: GeminiAgent) -> tuple[list[TreeNode], list[tuple[TreeNode, str]]]:
    """
    Generate multiple children for a list of nodes
    
    Args:
        nodes_with_code: List of parent nodes
        num_children: Number of children to generate per node
        agent: GeminiAgent instance
        
    Returns:
        List of children nodes and errors if any
    """
    all_children = []
    errors = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        
        for node in nodes_with_code:
            # Determine if this is a counting task by checking the code
            # Check if node has curr_code attribute, otherwise use the first code from codelist
            if hasattr(node, 'curr_code'):
                node_code = node.curr_code.lower()
            else:
                # Get the code from the node's codelist if available
                node_code = node._[0].code.lower() if node._ and len(node._) > 0 else ""
                
            for i in range(num_children):
                suggestive_hint = f"Improve the code (approach {i+1})"
                futures.append(executor.submit(generate_one_child, node, suggestive_hint, agent))
        
        for future in as_completed(futures):
            child, error = future.result()
            if error:
                parent = child  # If error, child is actually the parent
                errors.append((parent, error))
            else:
                all_children.append(child)
    
    return all_children, errors

def evaluate(task: Task, agent: GeminiAgent) -> List[Tuple[TreeNode, str]]:
    """
    Evaluate a task using the Gemini agent
    
    Args:
        task: Task to evaluate
        agent: GeminiAgent instance
        
    Returns:
        List of (node, evaluation) pairs
    """
    # Initialize with root node
    root = root_factory(task)
    
    # Generate branches up to MAX_DEPTH
    curr_nodes = [root]
    for depth in range(MAX_DEPTH):
        children, errors = generate_children(curr_nodes, BRANCH_WIDTH, agent)
        if errors:
            logger.warning(f"Errors at depth {depth}: {len(errors)}")
        
        if not children:
            break
        
        curr_nodes = children
    
    # Evaluate the final nodes
    evaluations = []
    
    for node in curr_nodes:
        # Construct evaluation prompt
        prompt = (
            f"Question: {task.question}\n\n"
            f"Proposed solution:\n```python\n{node.curr_code}\n```\n\n"
            "Evaluate this solution. Is it correct and efficient? What are its strengths and weaknesses?"
        )
        
        # Get evaluation
        images = None
        if hasattr(task, 'images') and task.images:
            images = task.images
        
        # Use the orchestrated approach for evaluation
        evaluation = agent.generate_orchestrated(prompt, images)
        evaluations.append((node, evaluation))
    
    return evaluations

def eval_a_batch(batch: Iterator[FullTask], agent: GeminiAgent = None, use_orchestration: bool = True) -> Tuple[int, int]:
    """
    Evaluate a batch of tasks
    
    Args:
        batch: Iterator of tasks to evaluate
        agent: GeminiAgent instance (created if None)
        use_orchestration: Whether to use the orchestrated approach
        
    Returns:
        Tuple of (correct count, total count)
    """
    if agent is None:
        agent = GeminiAgent(
            use_depth=True,
            use_segmentation=True,
            use_novel_view=True,
            use_point_cloud=True,
            verbose=False
        )
    
    correct = 0
    total = 0
    
    for fulltask in batch:
        # Convert FullTask to Task
        task = fulltask_to_task(fulltask)
        
        # Run evaluation
        evaluations = evaluate(task, agent)
        
        # Check if any evaluation contains the correct answer
        answer = fulltask['answer']
        outputs = [eval_text for _, eval_text in evaluations]
        
        if judge_if_any(outputs, answer):
            correct += 1
        
        total += 1
        
        logger.info(f"Progress: {correct}/{total} correct")
    
    return correct, total

def fulltask_to_task(fulltask: FullTask) -> Task:
    """Convert FullTask to Task"""
    from coc.util.misc import fulltask_to_task
    return fulltask_to_task(fulltask)

def judge_if_any(outputs: List[str], answer: str) -> bool:
    """Check if any output contains the correct answer"""
    from coc.util.misc import judge_if_any
    return judge_if_any(outputs, answer)

def main_eval_muir(partition: str = "Counting", use_orchestration: bool = True):
    """Run evaluation on MUIR dataset
    
    Args:
        partition: Dataset partition to evaluate (Counting or Ordering)
        use_orchestration: Whether to use the orchestrated approach
    """
    from coc.data.muir import muir
    
    # Initialize agent with all 3D vision capabilities
    agent = GeminiAgent(
        use_depth=True,
        use_segmentation=True,
        use_novel_view=True,
        use_point_cloud=True,
        verbose=False
    )
    
    # Load dataset and evaluate
    batch = muir(partition)
    correct, total = eval_a_batch(batch, agent, use_orchestration)
    
    success_rate = correct / total if total > 0 else 0
    print(f"Success rate on {partition}: {success_rate:.2%} ({correct}/{total})")
    
    return success_rate

if __name__ == "__main__":
    main_eval_muir()