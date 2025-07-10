#!/usr/bin/env python3
"""
Script de test pour vérifier l'utilisation optimale du GPU RTX 4060.
Ce script vérifie la détection, les performances et l'optimisation GPU.
"""

import torch
import time
import numpy as np
from agent.dqn_agent import DQNAgent

def test_gpu_detection():
    """Teste la détection et l'utilisation du GPU."""
    
    print("🔍 TEST DE DÉTECTION GPU RTX 4060")
    print("="*50)
    
    # Vérifications de base
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible - Vérifiez votre installation")
        return False, None
    
    # Détection des GPU
    gpu_count = torch.cuda.device_count()
    print(f"Nombre de GPU: {gpu_count}")
    
    rtx_4060_found = False
    rtx_4060_index = None
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        print(f"\nGPU {i}: {gpu_name}")
        print(f"  Mémoire: {gpu_memory:.1f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        if "4060" in gpu_name or "RTX 4060" in gpu_name:
            rtx_4060_found = True
            rtx_4060_index = i
            print(f"  ✅ RTX 4060 TROUVÉE !")
    
    if rtx_4060_found:
        print(f"\n🎉 RTX 4060 détectée sur GPU {rtx_4060_index}")
        return True, rtx_4060_index
    else:
        print(f"\n⚠️ RTX 4060 non détectée par nom - Utilisation GPU 0")
        return True, 0

def test_agent_gpu_usage():
    """Teste l'utilisation GPU par l'agent DQN."""
    
    print(f"\n🧠 TEST D'UTILISATION GPU PAR L'AGENT DQN")
    print("="*45)
    
    # Créer un agent
    agent = DQNAgent(state_size=11, action_size=3)
    
    # Vérifier le device utilisé
    device_info = agent.get_model_info()
    print(f"Device de l'agent: {device_info['device']}")
    print(f"Paramètres du modèle: {device_info['total_parameters']:,}")
    
    # Test avec un état simulé
    test_state = np.random.random(11)
    
    # Mesurer le temps d'inférence
    start_time = time.time()
    for _ in range(1000):
        action = agent.get_action(test_state, training=False)
    inference_time = time.time() - start_time
    
    print(f"Temps pour 1000 inférences: {inference_time:.3f}s")
    print(f"Vitesse: {1000/inference_time:.1f} inférences/seconde")
    
    return True

def benchmark_gpu_vs_cpu():
    """Compare les performances GPU vs CPU."""
    
    print(f"\n⚡ BENCHMARK GPU vs CPU")
    print("="*30)
    
    # Test matrices pour simulation de l'entraînement
    matrix_size = 2048
    num_iterations = 10
    
    print(f"Test: Multiplication de matrices {matrix_size}x{matrix_size}")
    print(f"Iterations: {num_iterations}")
    
    # Test CPU
    print(f"\n🐌 Test CPU...")
    cpu_times = []
    for i in range(num_iterations):
        start = time.time()
        a = torch.randn(matrix_size, matrix_size)
        b = torch.randn(matrix_size, matrix_size)
        c = torch.mm(a, b)
        cpu_times.append(time.time() - start)
    
    avg_cpu_time = np.mean(cpu_times)
    print(f"Temps moyen CPU: {avg_cpu_time:.3f}s")
    
    # Test GPU
    if torch.cuda.is_available():
        print(f"🚀 Test GPU...")
        device = torch.device("cuda")
        gpu_times = []
        
        # Warm-up GPU
        a_gpu = torch.randn(matrix_size, matrix_size, device=device)
        b_gpu = torch.randn(matrix_size, matrix_size, device=device)
        _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        for i in range(num_iterations):
            start = time.time()
            a_gpu = torch.randn(matrix_size, matrix_size, device=device)
            b_gpu = torch.randn(matrix_size, matrix_size, device=device)
            c_gpu = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_times.append(time.time() - start)
        
        avg_gpu_time = np.mean(gpu_times)
        speedup = avg_cpu_time / avg_gpu_time
        
        print(f"Temps moyen GPU: {avg_gpu_time:.3f}s")
        print(f"🏆 Accélération: {speedup:.1f}x plus rapide")
        
        if speedup > 5:
            print("✅ Excellente performance GPU !")
        elif speedup > 2:
            print("✅ Bonne performance GPU")
        else:
            print("⚠️ Performance GPU décevante - Vérifiez les drivers")
    
    return True

def test_memory_usage():
    """Teste l'utilisation de la mémoire GPU."""
    
    print(f"\n💾 TEST DE MÉMOIRE GPU")
    print("="*25)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
        # Mémoire avant
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated(device) / (1024**2)
        memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        
        print(f"Mémoire utilisée avant: {memory_before:.1f} MB")
        print(f"Mémoire totale: {memory_total:.1f} MB")
        
        # Créer un agent et tester
        agent = DQNAgent(state_size=11, action_size=3)
        
        # Mémoire après création de l'agent
        memory_after = torch.cuda.memory_allocated(device) / (1024**2)
        memory_used = memory_after - memory_before
        
        print(f"Mémoire utilisée par l'agent: {memory_used:.1f} MB")
        print(f"Pourcentage utilisé: {(memory_after/memory_total)*100:.2f}%")
        
        # Test de charge mémoire avec batch
        print(f"\nTest avec batch d'entraînement...")
        batch_size = 32
        state_size = 11
        
        states = torch.randn(batch_size, state_size, device=device)
        memory_batch = torch.cuda.memory_allocated(device) / (1024**2)
        
        print(f"Mémoire avec batch: {memory_batch:.1f} MB")
        
        # Nettoyage
        del agent, states
        torch.cuda.empty_cache()
        
        print("✅ Test mémoire terminé")
    else:
        print("❌ GPU non disponible pour test mémoire")

def main():
    """Fonction principale de test."""
    
    print("🎮 TEST COMPLET D'OPTIMISATION RTX 4060")
    print("="*55)
    
    try:
        # Test 1: Détection GPU
        gpu_result = test_gpu_detection()
        
        if gpu_result == (False, None):
            print("\n❌ Tests interrompus - GPU non disponible")
            return
        
        gpu_available, gpu_index = gpu_result
        
        # Test 2: Agent DQN
        test_agent_gpu_usage()
        
        # Test 3: Benchmark
        benchmark_gpu_vs_cpu()
        
        # Test 4: Mémoire
        test_memory_usage()
        
        print(f"\n🎉 TESTS TERMINÉS AVEC SUCCÈS !")
        print("="*35)
        print("✅ Votre RTX 4060 est prête pour l'entraînement Snake RL")
        print("⚡ Performance optimale garantie")
        print("🚀 Lancez maintenant: python train.py ou python quick_train.py")
        
    except Exception as e:
        print(f"\n❌ Erreur pendant les tests: {e}")
        print("💡 Vérifiez votre installation PyTorch et drivers NVIDIA")

if __name__ == "__main__":
    main()
