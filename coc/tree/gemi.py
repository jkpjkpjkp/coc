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

# Import 3D vision modules if available - these will be optional
try:
    # Try direct import first (using the specific paths in this codebase)
    from coc.tool.depth import DepthAnything
except ImportError:
    try:
        # Try importing from submodules if not in coc.tool
        from submodules.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnything
    except ImportError:
        logging.warning("DepthAnything module could not be imported")
        DepthAnything = None

try:
    from coc.tool._3d import Fast3R
except ImportError:
    try:
        from submodules.fast3r.fast3r.fast3r import Fast3R
    except ImportError:
        logging.warning("Fast3R module could not be imported")
        Fast3R = None

try:
    from coc.tool._3d import PoinTr
except ImportError:
    try:
        from submodules.PoinTr.models.pointr import PoinTr
    except ImportError:
        logging.warning("PoinTr module could not be imported")
        PoinTr = None

try:
    from coc.tool.sam import sam_model_registry
except ImportError:
    try:
        from submodules.sam2.sam2.build_sam import sam_model_registry
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
    """
    
    def __init__(self, 
                 use_depth: bool = True,
                 use_segmentation: bool = True,
                 use_novel_view: bool = True,
                 use_point_cloud: bool = True):
        """
        Initialize the Gemini agent with 3D vision capabilities
        
        Args:
            use_depth: Whether to use depth estimation
            use_segmentation: Whether to use segmentation
            use_novel_view: Whether to use novel view synthesis
            use_point_cloud: Whether to use point cloud processing
        """
        # Initialize Gemini VQA tool
        self.gemini = Gemini()
        
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
        
        return (
            "You are a helpful multimodal assistant with advanced 3D vision capabilities. "
            f"You have access to: {', '.join(capabilities)}. "
            "Use this information to better understand spatial relationships "
            "and 3D structure in images."
        )
    
    def _get_headers(self):
        """Get headers for the API request"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _encode_image(self, image):
        """Convert PIL Image to base64 string"""
        if isinstance(image, str):
            return image
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _prepare_messages(self, prompt, images=None, enhanced_data=None):
        """
        Prepare messages with text, images, and enhanced vision data
        
        Args:
            prompt: Text prompt
            images: List of PIL Images
            enhanced_data: Dictionary of enhanced vision data
            
        Returns:
            Formatted messages for API request
        """
        # System message with enhanced capabilities description
        system_message = self._get_prompt_capabilities()
        
        messages = [{"role": "system", "content": system_message}]
        
        # Create user message content
        if images or enhanced_data:
            content = [{"type": "text", "text": prompt}]
            
            # Add original images
            if images:
                for img in images:
                    encoded_img = self._encode_image(img)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_img}"}
                    })
            
            # Add enhanced vision data (depth maps, novel views, etc.)
            if enhanced_data:
                if "depth_maps" in enhanced_data:
                    for depth_map in enhanced_data["depth_maps"]:
                        encoded_img = self._encode_image(depth_map)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_img}"}
                        })
                
                if "novel_views" in enhanced_data:
                    for novel_view in enhanced_data["novel_views"]:
                        encoded_img = self._encode_image(novel_view)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_img}"}
                        })
                
                # Add textual descriptions of 3D data
                if "point_cloud_summary" in enhanced_data:
                    content.append({
                        "type": "text",
                        "text": f"3D Point Cloud Analysis: {enhanced_data['point_cloud_summary']}"
                    })
                
                if "segmentation_summary" in enhanced_data:
                    content.append({
                        "type": "text",
                        "text": f"Segmentation Analysis: {enhanced_data['segmentation_summary']}"
                    })
            
            messages.append({"role": "user", "content": content})
        else:
            # Text-only message
            messages.append({"role": "user", "content": prompt})
        
        return messages
    
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
        f"Given the following code:\n```python\n{parent.curr_code}\n```\n"
        f"Your task: {suggestive_hint}\n"
        "Generate improved code that addresses this task. "
        "The code should be complete, well-structured, and functional."
    )
    
    # Get response
    try:
        # Prepare images if available
        images = None
        if hasattr(parent, 'images') and parent.images:
            images = parent.images
        
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
        
        evaluation = agent.generate(prompt, images)
        evaluations.append((node, evaluation))
    
    return evaluations

def eval_a_batch(batch: Iterator[FullTask], agent: GeminiAgent = None) -> Tuple[int, int]:
    """
    Evaluate a batch of tasks
    
    Args:
        batch: Iterator of tasks to evaluate
        agent: GeminiAgent instance (created if None)
        
    Returns:
        Tuple of (correct count, total count)
    """
    if agent is None:
        agent = GeminiAgent()
    
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

def main_eval_muir(partition: str = "Counting"):
    """Run evaluation on MUIR dataset"""
    from coc.data.muir import muir
    
    # Initialize agent with all 3D vision capabilities
    agent = GeminiAgent(
        use_depth=True,
        use_segmentation=True,
        use_novel_view=True,
        use_point_cloud=True
    )
    
    # Load dataset and evaluate
    batch = muir(partition)
    correct, total = eval_a_batch(batch, agent)
    
    success_rate = correct / total if total > 0 else 0
    print(f"Success rate on {partition}: {success_rate:.2%} ({correct}/{total})")
    
    return success_rate

if __name__ == "__main__":
    main_eval_muir() 