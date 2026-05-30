#!/usr/bin/env python3
"""Probe Mesa Xe support for VK_KHR_global_priority on this Arc.

Two checks:
1. Enumerate per-queue-family supported priority list via
   VkQueueFamilyGlobalPriorityPropertiesKHR pNext on
   vkGetPhysicalDeviceQueueFamilyProperties2.
2. Try to actually CREATE a device + queue at each priority. Some
   priorities (REALTIME especially) may require CAP_SYS_NICE, in
   which case create returns VK_ERROR_NOT_PERMITTED_KHR.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / 'viz_authoring/src'))

import vulkan as vk
from viz_authoring.vk.context import VkContext


PRIORITY_NAMES = {
    vk.VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR:      'LOW',
    vk.VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR:   'MEDIUM',
    vk.VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR:     'HIGH',
    vk.VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR: 'REALTIME',
}


def query_supported_priorities(physical_device):
    """Returns {queue_family_index: [priority_value, ...]}"""
    # Get count
    qf_props = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    n = len(qf_props)

    # Build pNext-chained VkQueueFamilyProperties2 array
    gp_structs = [
        vk.VkQueueFamilyGlobalPriorityPropertiesKHR(
            sType=vk.VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR,
            priorityCount=0,
        )
        for _ in range(n)
    ]
    qfp2 = [
        vk.VkQueueFamilyProperties2(
            sType=vk.VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
            pNext=gp_structs[i],
        )
        for i in range(n)
    ]
    # The high-level wrapper may not chain pNext for the array form;
    # instead make individual calls with count=1 isn't supported either.
    # Try the array call and see what we get.
    result = vk.vkGetPhysicalDeviceQueueFamilyProperties2(physical_device,
                                                            qfp2)
    out = {}
    for i in range(n):
        gp = gp_structs[i]
        # priorities is an array; only first priorityCount are valid
        out[i] = [gp.priorities[j] for j in range(gp.priorityCount)]
    return out


def try_create_with_priority(physical_device, queue_family: int,
                                priority_value: int) -> str:
    """Try to vkCreateDevice with a queue at the given priority.
    Returns 'OK' on success or an error description."""
    gp_create = vk.VkDeviceQueueGlobalPriorityCreateInfoKHR(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
        globalPriority=priority_value,
    )
    qci = vk.VkDeviceQueueCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        pNext=gp_create,
        queueFamilyIndex=queue_family,
        queueCount=1,
        pQueuePriorities=[1.0],
    )
    dci = vk.VkDeviceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        queueCreateInfoCount=1,
        pQueueCreateInfos=[qci],
        enabledExtensionCount=1,
        ppEnabledExtensionNames=[vk.VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME],
    )
    try:
        device = vk.vkCreateDevice(physical_device, dci, None)
        vk.vkDestroyDevice(device, None)
        return 'OK'
    except vk.VkException as e:
        # vulkan-py raises subclasses for known error codes
        name = type(e).__name__
        return f'{name}'
    except Exception as e:
        return f'{type(e).__name__}: {e}'


def main():
    ctx = VkContext(instance_extensions=[], pipeline_cache_path=None)
    ctx.select_device()
    print(f'device: {ctx.device_name}')
    print()

    print('=== queue families ===')
    qf_props = vk.vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device)
    for i, qf in enumerate(qf_props):
        flags = []
        if qf.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT: flags.append('GRAPHICS')
        if qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:  flags.append('COMPUTE')
        if qf.queueFlags & vk.VK_QUEUE_TRANSFER_BIT: flags.append('TRANSFER')
        print(f'  family {i}: count={qf.queueCount} flags={"|".join(flags)}')

    print()
    print('=== per-family priority list (via query extension) ===')
    try:
        supported = query_supported_priorities(ctx.physical_device)
        for fam, prios in supported.items():
            names = [PRIORITY_NAMES.get(p, f'?{p}') for p in prios]
            if names:
                print(f'  family {fam}: {", ".join(names)}')
            else:
                print(f'  family {fam}: (none / wrapper did not populate)')
    except Exception as e:
        print(f'  query failed: {e}')

    print()
    print('=== actual vkCreateDevice attempts on queue family 0 ===')
    for name, value in [
        ('LOW',      vk.VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR),
        ('MEDIUM',   vk.VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR),
        ('HIGH',     vk.VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR),
        ('REALTIME', vk.VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR),
    ]:
        outcome = try_create_with_priority(ctx.physical_device, 0, value)
        print(f'  {name:10s}: {outcome}')


if __name__ == '__main__':
    main()
