'''
Global processes manager

Enables expensive processes used in one place to be reused elsewhere.
For example: LlamaCPP server shared between T2T operation instances.
'''

import logging
from enum import Enum

from utils.helpers.singleton import Singleton

from .error import UnknownProcessError, UnloadedProcessError

class ProcessType(Enum):
    LLAMACPP = "llamacpp"
    SHERPA = "sherpa_stt"
    HW_MIC = "hw_mic"
    DISCORD = "discord"

class ProcessManager(metaclass=Singleton):
    loaded_processes = dict()
    
    '''Perform initial load'''
    async def load(self, process_type: ProcessType, process_config: dict | None = None):
        logging.info("Loading process by type {}".format(process_type.value))
        match process_type:
            case ProcessType.LLAMACPP:
                from .drivers.llamacpp import LlamaCPPProcess
                self.loaded_processes[ProcessType.LLAMACPP] = LlamaCPPProcess()
                if process_config is not None:
                    self.loaded_processes[ProcessType.LLAMACPP].set_runtime_config(process_config)
                await self.loaded_processes[ProcessType.LLAMACPP].reload()
            case ProcessType.SHERPA:
                from .drivers.sherpa_server import SherpaSTTProcess
                self.loaded_processes[ProcessType.SHERPA] = SherpaSTTProcess()
                if process_config is not None:
                    self.loaded_processes[ProcessType.SHERPA].set_runtime_config(process_config)
                await self.loaded_processes[ProcessType.SHERPA].reload()
            case ProcessType.HW_MIC:
                from .drivers.hw_mic import HwMicProcess
                self.loaded_processes[ProcessType.HW_MIC] = HwMicProcess()
                if process_config is not None:
                    self.loaded_processes[ProcessType.HW_MIC].set_runtime_config(process_config)
                await self.loaded_processes[ProcessType.HW_MIC].reload()
            case ProcessType.DISCORD:
                from .drivers.discord import DiscordProcess
                self.loaded_processes[ProcessType.DISCORD] = DiscordProcess()
                if process_config is not None:
                    self.loaded_processes[ProcessType.DISCORD].set_runtime_config(process_config)
                await self.loaded_processes[ProcessType.DISCORD].reload()
            case _:
                raise UnknownProcessError(process_type)
        
    '''Reload any process where reload_signal is True'''
    async def reload(self):
        for process_type in self.loaded_processes:
            if self.loaded_processes[process_type] and self.loaded_processes[process_type].reload_signal:
                logging.info("Reloading process {}".format(self.loaded_processes[process_type].id))
                await self.loaded_processes[process_type].reload()
        
    '''Unload any process where unload_signal is True'''
    async def unload(self):
        for process_type in self.loaded_processes:
            if self.loaded_processes[process_type] and self.loaded_processes[process_type].unload_signal:
                logging.info("Unloading process {}".format(self.loaded_processes[process_type].id))
                await self.loaded_processes[process_type].unload()
                
    async def link(self, link_id: str, process_type: ProcessType, process_config: dict | None = None):
        if not (process_type in self.loaded_processes and self.loaded_processes[process_type]):
            await self.load(process_type, process_config=process_config)
        elif process_config is not None:
            self.loaded_processes[process_type].set_runtime_config(process_config)
            if self.loaded_processes[process_type].process is None:
                await self.loaded_processes[process_type].reload()
            
        await self.loaded_processes[process_type].link(link_id, process_config=process_config)
        
    async def unlink(self, link_id: str, process_type: ProcessType):
        if not (process_type in self.loaded_processes and self.loaded_processes[process_type]):
            raise UnloadedProcessError(process_type.value)
            
        await self.loaded_processes[process_type].unlink(link_id)
    
    def signal_reload(self, process_type: ProcessType):
        if not (process_type in self.loaded_processes and self.loaded_processes[process_type]):
            raise UnloadedProcessError(process_type.value)
            
        self.loaded_processes[process_type].reload_signal = True
        
    def signal_unload(self, process_type: ProcessType):
        if not (process_type in self.loaded_processes and self.loaded_processes[process_type]):
            raise UnloadedProcessError(process_type.value)
            
        self.loaded_processes[process_type].unload_signal = True
        
    def get_process(self, process_type: ProcessType):
        if not (process_type in self.loaded_processes and self.loaded_processes[process_type]):
            raise UnloadedProcessError(process_type.value)
            
        return self.loaded_processes[process_type]
