"""
GPU Scheduler Server for Stable Diffusion Generation

This server manages a queue of generation requests and schedules them
on available GPUs that have no active processes.
"""

import os
import json
import time
import base64
import subprocess
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from queue import Queue, Empty
from threading import Thread, Lock
import logging
from datetime import datetime

# from flask import Flask, request, jsonify

# Import from stable_diff.py
from stable_diff import PromptConfig, profile_stable_diffusion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# app = Flask(__name__)

@dataclass
class GenerationRequest:
    """Request for image generation"""
    request_id: str
    prompt: str
    negative_prompt: str = ""
    use_negative_prompt: bool = True
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = 42
    height: int = 512
    width: int = 512
    use_fp16: bool = True
    scheduler_name: str = "default"
    
    def to_dict(self):
        return asdict(self)

@dataclass
class GenerationResult:
    """Result from image generation"""
    request_id: str
    success: bool
    config: Dict[str, Any]
    timings: Dict[str, float]
    image_base64: Optional[str] = None
    error: Optional[str] = None
    gpu_id: Optional[int] = None
    
    def to_dict(self):
        return asdict(self)

class GPUManager:
    """Manages GPU availability and allocation"""
    
    def __init__(self):
        self.gpu_lock = Lock()
        self.allocated_gpus = set()
    
    def get_available_gpus(self) -> List[int]:
        """
        Get list of GPUs with no active processes.
        Returns list of GPU IDs that are completely idle.
        """
        try:
            # Run nvidia-smi to get GPU utilization and process info
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpu_stats = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    gpu_id = int(parts[0])
                    gpu_util = int(parts[1])
                    mem_used = int(parts[2])
                    gpu_stats.append((gpu_id, gpu_util, mem_used))
            
            # Check for processes on each GPU
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', 
                 '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get GPU UUIDs for mapping
            uuid_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,gpu_uuid', 
                 '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            
            uuid_to_id = {}
            for line in uuid_result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    uuid_to_id[parts[1]] = int(parts[0])
            
            # Track which GPUs have processes
            gpus_with_processes = set()
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 1:
                        gpu_uuid = parts[0]
                        if gpu_uuid in uuid_to_id:
                            gpus_with_processes.add(uuid_to_id[gpu_uuid])
            
            # Find completely idle GPUs (no processes and low utilization)
            available = []
            for gpu_id, gpu_util, mem_used in gpu_stats:
                if (gpu_id not in gpus_with_processes and 
                    gpu_id not in self.allocated_gpus and
                    gpu_util < 5 and  # Less than 5% GPU utilization
                    mem_used < 100):   # Less than 100MB memory used
                    available.append(gpu_id)
            
            logger.info(f"Available GPUs: {available}")
            return available
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error querying nvidia-smi: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in get_available_gpus: {e}")
            return []
    
    def allocate_gpu(self) -> Optional[int]:
        """Allocate an available GPU. Returns GPU ID or None if no GPU available."""
        with self.gpu_lock:
            available = self.get_available_gpus()
            if available:
                gpu_id = available[0]
                self.allocated_gpus.add(gpu_id)
                logger.info(f"Allocated GPU {gpu_id}")
                return gpu_id
            return None
    
    def release_gpu(self, gpu_id: int):
        """Release a GPU back to the available pool."""
        with self.gpu_lock:
            if gpu_id in self.allocated_gpus:
                self.allocated_gpus.discard(gpu_id)
                logger.info(f"Released GPU {gpu_id}")

class GenerationWorker(Thread):
    """Worker thread that processes generation requests on assigned GPU"""
    
    def __init__(self, request_queue: Queue, result_queue: Queue, 
                 gpu_manager: GPUManager):
        super().__init__(daemon=True)
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.gpu_manager = gpu_manager
        self.running = True
    
    def run(self):
        """Main worker loop"""
        logger.info(f"Worker thread {self.name} started")
        
        while self.running:
            try:
                # Wait for a request (with timeout to allow checking running flag)
                try:
                    gen_request = self.request_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                logger.info(f"Processing request {gen_request.request_id}")
                
                # Wait for an available GPU
                gpu_id = None
                while gpu_id is None and self.running:
                    gpu_id = self.gpu_manager.allocate_gpu()
                    if gpu_id is None:
                        logger.info("No GPU available, waiting...")
                        time.sleep(2)
                
                if not self.running:
                    break
                
                # Process the request
                try:
                    result = self.generate_image(gen_request, gpu_id)
                except Exception as e:
                    logger.error(f"Error generating image: {e}", exc_info=True)
                    result = GenerationResult(
                        request_id=gen_request.request_id,
                        success=False,
                        config=gen_request.to_dict(),
                        timings={},
                        error=str(e),
                        gpu_id=gpu_id
                    )
                finally:
                    # Always release the GPU
                    self.gpu_manager.release_gpu(gpu_id)
                
                # Put result in queue
                self.result_queue.put(result)
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
    
    def generate_image(self, gen_request: GenerationRequest, 
                      gpu_id: int) -> GenerationResult:
        """Generate image using the assigned GPU"""
        import os
        from io import BytesIO
        import base64
        
        try:
            # Set the GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            logger.info(f"Generating image on GPU {gpu_id}")
            
            # Create PromptConfig from GenerationRequest
            config = PromptConfig(
                prompt=gen_request.prompt,
                negative_prompt=gen_request.negative_prompt,
                use_negative_prompt=gen_request.use_negative_prompt,
                num_inference_steps=gen_request.num_inference_steps,
                guidance_scale=gen_request.guidance_scale,
                seed=gen_request.seed,
                height=gen_request.height,
                width=gen_request.width,
                use_fp16=gen_request.use_fp16,
                scheduler_name=gen_request.scheduler_name,
                output_filename=f"temp_{gen_request.request_id}.png"
            )
            
            # Call the profiling function from stable_diff.py
            image_pil, timings = profile_stable_diffusion(config)
            
            # Convert to base64
            buffered = BytesIO()
            image_pil.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Clean up temp file
            if os.path.exists(config.output_filename):
                os.remove(config.output_filename)
            
            logger.info(f"Successfully generated image for request {gen_request.request_id}")
            
            return GenerationResult(
                request_id=gen_request.request_id,
                success=True,
                config=gen_request.to_dict(),
                timings=timings,
                image_base64=image_base64,
                gpu_id=gpu_id
            )
            
        except Exception as e:
            logger.error(f"Error in generate_image: {e}", exc_info=True)
            raise

class GenerationServer:
    """Main server for managing generation requests"""
    
    def __init__(self, num_workers: int = 4):
        self.request_queue = Queue()
        self.result_queue = Queue()
        self.gpu_manager = GPUManager()
        self.results_cache = {}
        self.cache_lock = Lock()
        
        # Start worker threads
        self.workers = []
        for i in range(num_workers):
            worker = GenerationWorker(
                self.request_queue, 
                self.result_queue, 
                self.gpu_manager
            )
            worker.start()
            self.workers.append(worker)
        
        # Start result collector thread
        self.result_collector = Thread(target=self._collect_results, daemon=True)
        self.result_collector.start()
        
        logger.info(f"Server started with {num_workers} workers")
    
    def _collect_results(self):
        """Collect results from result queue and cache them"""
        while True:
            try:
                result = self.result_queue.get(timeout=1.0)
                with self.cache_lock:
                    self.results_cache[result.request_id] = result
                logger.info(f"Cached result for request {result.request_id}")
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error collecting results: {e}", exc_info=True)
    
    def submit_request(self, gen_request: GenerationRequest) -> str:
        """Submit a generation request and return request ID"""
        self.request_queue.put(gen_request)
        logger.info(f"Submitted request {gen_request.request_id}")
        return gen_request.request_id
    
    def get_result(self, request_id: str) -> Optional[GenerationResult]:
        """Get result for a request ID if available"""
        with self.cache_lock:
            return self.results_cache.get(request_id)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "pending_requests": self.request_queue.qsize(),
            "available_gpus": len(self.gpu_manager.get_available_gpus()),
            "allocated_gpus": len(self.gpu_manager.allocated_gpus),
            "cached_results": len(self.results_cache)
        }

n = GPUManager()
print(n.get_available_gpus())


# # Initialize server
# generation_server = GenerationServer(num_workers=4)

# @app.route('/generate', methods=['POST'])
# def generate():
#     """Submit a new generation request"""
#     try:
#         data = request.json
        
#         # Generate unique request ID
#         request_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(data)}"
        
#         # Create generation request
#         gen_request = GenerationRequest(
#             request_id=request_id,
#             prompt=data['prompt'],
#             negative_prompt=data.get('negative_prompt', ''),
#             use_negative_prompt=data.get('use_negative_prompt', True),
#             num_inference_steps=data.get('num_inference_steps', 50),
#             guidance_scale=data.get('guidance_scale', 7.5),
#             seed=data.get('seed', 42),
#             height=data.get('height', 512),
#             width=data.get('width', 512),
#             use_fp16=data.get('use_fp16', True),
#             scheduler_name=data.get('scheduler_name', 'default')
#         )
        
#         # Submit request
#         generation_server.submit_request(gen_request)
        
#         return jsonify({
#             "success": True,
#             "request_id": request_id,
#             "message": "Request submitted successfully"
#         }), 202
        
#     except Exception as e:
#         logger.error(f"Error in /generate: {e}", exc_info=True)
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 400

# @app.route('/result/<request_id>', methods=['GET'])
# def get_result(request_id: str):
#     """Get result for a specific request ID"""
#     try:
#         result = generation_server.get_result(request_id)
        
#         if result is None:
#             return jsonify({
#                 "success": False,
#                 "status": "pending",
#                 "message": "Result not yet available"
#             }), 202
        
#         return jsonify(result.to_dict()), 200
        
#     except Exception as e:
#         logger.error(f"Error in /result: {e}", exc_info=True)
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500

# @app.route('/status', methods=['GET'])
# def status():
#     """Get server status"""
#     try:
#         return jsonify({
#             "success": True,
#             "status": generation_server.get_queue_status()
#         }), 200
#     except Exception as e:
#         logger.error(f"Error in /status: {e}", exc_info=True)
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500

# @app.route('/health', methods=['GET'])
# def health():
#     """Health check endpoint"""
#     return jsonify({"status": "healthy"}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, threaded=True)