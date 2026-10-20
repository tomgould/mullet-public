import pytest
import threading
from mullet.core.tools.process_registry import ProcessRegistry

@pytest.fixture
def registry():
    reg = ProcessRegistry()
    reg.cleanup_all() # Ensure a clean state for each test
    return reg

def test_register_and_list(registry):
    registry.register(1, "test_process_1")
    active_processes = registry.list_active()
    assert len(active_processes) == 1
    assert active_processes[0][0] == 1
    assert active_processes[0][1] == "test_process_1"

def test_unregister(registry):
    registry.register(2, "test_process_2")
    registry.unregister(2)
    active_processes = registry.list_active()
    assert len(active_processes) == 0

def test_unregister_non_existent_pid(registry):
    registry.unregister(999) # Should not raise an error
    active_processes = registry.list_active()
    assert len(active_processes) == 0

def test_cleanup_all(registry):
    registry.register(3, "test_process_3")
    registry.register(4, "test_process_4")
    registry.cleanup_all()
    active_processes = registry.list_active()
    assert len(active_processes) == 0

def test_thread_safety_registration(registry):
    num_threads = 10
    num_registrations_per_thread = 10
    threads = []

    def register_processes(start_pid):
        for i in range(num_registrations_per_thread):
            registry.register(start_pid + i, f"thread_process_{start_pid + i}")

    for i in range(num_threads):
        thread = threading.Thread(target=register_processes, args=(i * num_registrations_per_thread,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    active_processes = registry.list_active()
    assert len(active_processes) == num_threads * num_registrations_per_thread

def test_thread_safety_unregister(registry):
    # Register some initial processes
    for i in range(20):
        registry.register(i, f"initial_process_{i}")

    num_threads = 5
    threads = []

    def unregister_processes(start_pid):
        for i in range(start_pid, start_pid + 4): # Unregister 4 processes per thread
            registry.unregister(i)

    for i in range(num_threads):
        thread = threading.Thread(target=unregister_processes, args=(i * 4,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    active_processes = registry.list_active()
    assert len(active_processes) == 0 # All 20 processes should be unregistered

def test_multiple_registrations_same_pid(registry):
    registry.register(10, "first_registration")
    registry.register(10, "second_registration") # Should overwrite or update
    active_processes = registry.list_active()
    assert len(active_processes) == 1
    assert active_processes[0][1] == "second_registration"

def test_list_active_empty(registry):
    active_processes = registry.list_active()
    assert len(active_processes) == 0
    assert isinstance(active_processes, list)
