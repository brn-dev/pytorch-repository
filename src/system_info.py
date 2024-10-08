import platform
import psutil
import subprocess
import re


def safe_del(d: dict, key: str):
    if key in d:
        del d[key]


# written by ChatGPT
# hopefully no security risks, too lazy to check


def get_nvidia_gpu_info():
    try:
        import pynvml
    except ImportError:
        print('Warning: pynvml module not installed! ')
        raise ImportError
    pynvml.nvmlInit()
    driver_version = pynvml.nvmlSystemGetDriverVersion()
    cuda_version = "Unknown"
    # Try to get the CUDA runtime version
    try:
        cuda_version_output = subprocess.check_output(['nvcc', '--version']).decode()
        # Parse the output to find the CUDA version
        match = re.search(r'release\s+(\d+\.\d+)', cuda_version_output)
        if match:
            cuda_version = match.group(1)
    except:
        pass

    device_count = pynvml.nvmlDeviceGetCount()
    gpu_info_list = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_total = memory_info.total // (1024 ** 2)  # Convert bytes to MB
        memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        # Get compute capability
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_capability = f"{major}.{minor}"
        except:
            compute_capability = "Unknown"
        gpu_info = {
            'name': name,
            'memory': f"{memory_total} MB",
            'memory_clock': f"{memory_clock} MHz",
            'compute_capability': compute_capability,
            'cuda_driver_version': driver_version,
            'cuda_runtime_version': cuda_version,
        }
        gpu_info_list.append(gpu_info)
    pynvml.nvmlShutdown()
    return gpu_info_list


def get_system_info(print_status: str = True, include_sensitive_stuff: bool = False):
    if print_status:
        print('Grabbing system information...', end='')

    info = {}
    # Basic platform info
    system = platform.system()
    info['platform'] = system
    info['platform_release'] = platform.release()
    info['platform_version'] = platform.version()
    info['architecture'] = platform.machine()
    info['hostname'] = platform.node()

    if system == 'Windows':
        try:
            import wmi
            c = wmi.WMI()

            # Processor info
            for processor in c.Win32_Processor():
                info['processor'] = {
                    'name': processor.Name.strip(),
                    'cores': processor.NumberOfCores,
                    'logical_cores': processor.NumberOfLogicalProcessors,
                    'speed': f"{processor.MaxClockSpeed} MHz"
                }

            # GPU info
            gpu_info_list = []
            for gpu in c.Win32_VideoController():
                gpu_info = {
                    'name': gpu.Name.strip(),
                    'driver_version': gpu.DriverVersion,
                    'video_processor': gpu.VideoProcessor.strip(),
                    'adapter_ram': f"{int(gpu.AdapterRAM) // (1024 ** 2)} MB" if gpu.AdapterRAM else 'Unknown',
                    'adapter_dac_type': gpu.AdapterDACType.strip(),
                    # 'video_mode_description': gpu.VideoModeDescription.strip(),
                    'manufacturer': gpu.AdapterCompatibility.strip(),
                }
                gpu_info_list.append(gpu_info)
            info['gpu'] = gpu_info_list

            # Check if GPU is from NVIDIA
            nvidia_present = any(
                'NVIDIA' in gpu['name'] or 'NVIDIA' in gpu.get('manufacturer', '') for gpu in gpu_info_list)

            if nvidia_present:
                # Use pynvml to get more detailed NVIDIA GPU info
                try:
                    nvidia_gpu_info_list = get_nvidia_gpu_info()
                    # Merge or replace existing NVIDIA GPU info
                    for i, gpu in enumerate(gpu_info_list):
                        if 'NVIDIA' in gpu['name'] or 'NVIDIA' in gpu.get('manufacturer', ''):
                            if i < len(nvidia_gpu_info_list):
                                gpu.update(nvidia_gpu_info_list[i])
                except Exception as e:
                    print(f"Error getting NVIDIA GPU info: {e}")

            # RAM speed
            ram_speeds = [int(mem.Speed) for mem in c.Win32_PhysicalMemory() if mem.Speed is not None]
            if ram_speeds:
                info['ram_speed'] = f"{max(ram_speeds)} MHz"
            else:
                info['ram_speed'] = "Unknown"
        except ImportError:
            print('Warning: wmi module not installed! ', end='')
            info['processor'] = 'Requires wmi module'
            info['gpu'] = [{'name': 'Requires wmi module', 'adapter_ram': 'Unknown'}]
            info['ram_speed'] = 'Requires wmi module'

    elif system == 'Linux':
        # Processor info
        try:
            lscpu_output = subprocess.check_output('lscpu', shell=True).decode()
            cpu_info = {}
            for line in lscpu_output.split('\n'):
                if line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key, value = parts
                        cpu_info[key.strip()] = value.strip()
            cores = int(cpu_info.get('Core(s) per socket', '0')) * int(cpu_info.get('Socket(s)', '1'))
            info['processor'] = {
                'name': cpu_info.get('Model name', 'Unknown'),
                'cores': cores,
                'logical_cores': int(cpu_info.get('CPU(s)', '0')),
                'speed': f"{cpu_info.get('CPU MHz', 'Unknown')} MHz"
            }
        except subprocess.CalledProcessError:
            info['processor'] = 'Unknown'
            info['processor_details'] = 'Unknown'

        # GPU info
        gpu_info_list = []
        try:
            lshw_output = subprocess.check_output('sudo lshw -C display', shell=True).decode()
            gpu_entries = lshw_output.strip().split('*-display')
            for entry in gpu_entries[1:]:
                gpu_info = {}
                product_match = re.search(r'product:\s*(.+)', entry)
                vendor_match = re.search(r'vendor:\s*(.+)', entry)
                memory_match = re.search(r'size:\s*(.+)', entry)

                gpu_info['name'] = product_match.group(1).strip() if product_match else 'Unknown'
                gpu_info['vendor'] = vendor_match.group(1).strip() if vendor_match else 'Unknown'
                gpu_info['memory'] = memory_match.group(1).strip() if memory_match else 'Unknown'
                gpu_info_list.append(gpu_info)
        except subprocess.CalledProcessError:
            gpu_info_list.append({'name': 'Unknown', 'vendor': 'Unknown', 'memory': 'Unknown'})

        # Check if GPU is from NVIDIA
        nvidia_present = any('NVIDIA' in gpu['vendor'] or 'NVIDIA' in gpu['name'] for gpu in gpu_info_list)
        if nvidia_present:
            try:
                import pynvml
                nvidia_gpu_info_list = get_nvidia_gpu_info()
                # Merge or replace existing NVIDIA GPU info
                for i, gpu in enumerate(gpu_info_list):
                    if 'NVIDIA' in gpu['name'] or 'NVIDIA' in gpu.get('vendor', ''):
                        if i < len(nvidia_gpu_info_list):
                            gpu.update(nvidia_gpu_info_list[i])
            except Exception as e:
                print(f"Error getting NVIDIA GPU info: {e}")

        info['gpu'] = gpu_info_list

        # RAM speed
        try:
            dmidecode_output = subprocess.check_output("sudo dmidecode -t memory", shell=True).decode()
            ram_speeds = re.findall(r'Speed:\s*(\d+)', dmidecode_output)
            if ram_speeds:
                info['ram_speed'] = f"{max(map(int, ram_speeds))} MHz"
            else:
                info['ram_speed'] = "Unknown"
        except subprocess.CalledProcessError:
            info['ram_speed'] = "Unknown"

    # RAM info (common for both)
    total_ram = psutil.virtual_memory().total
    info['ram'] = f"{round(total_ram / (1024.0 ** 3))} GB"

    if not include_sensitive_stuff:
        del info['hostname']
        del info['platform_version']

    for gpu in info['gpu']:
        safe_del(gpu, 'video_mode_description')
        if not include_sensitive_stuff:
            safe_del(gpu, 'driver_version')
            safe_del(gpu, 'cuda_driver_version')
            safe_del(gpu, 'cuda_runtime_version')

    if print_status:
        print(' done!')

    return info
