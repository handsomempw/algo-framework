"""输出管理器模块
处理算法结果的存储、转发等
"""
from typing import Dict, Any
import json
import logging
from pathlib import Path
from datetime import datetime

class OutputManager:
    """输出管理器"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'outputs'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def save_result(self, task_id: str, result: Dict[str, Any]):
        """保存算法处理结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{task_id}_{timestamp}.json"
            
            with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"结果已保存: {filename}")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")