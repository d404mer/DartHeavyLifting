"""
Диагностика маршрутизации DeckLink устройств
Проверяет конфигурацию входов/выходов и возможные дублирования
"""
import subprocess
import sys
import os

def check_ffmpeg_decklink():
    """Проверка поддержки DeckLink в ffmpeg"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "decklink", "-list_devices", "1", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "decklink" in result.stderr.lower() or "decklink" in result.stdout.lower():
            print("✅ FFmpeg поддерживает DeckLink")
            print("\nДоступные устройства DeckLink:")
            print(result.stderr)
            return True
        else:
            print("⚠️ FFmpeg не поддерживает DeckLink или устройства не найдены")
            print("Вывод:", result.stderr[:500])
            return False
    except FileNotFoundError:
        print("❌ FFmpeg не найден в PATH")
        return False
    except Exception as e:
        print(f"❌ Ошибка при проверке ffmpeg: {e}")
        return False

def check_blackmagic_desktop_video():
    """Проверка наличия Blackmagic Desktop Video"""
    paths_to_check = [
        r"C:\Program Files\Blackmagic Design\Blackmagic Desktop Video",
        r"C:\Program Files (x86)\Blackmagic Design\Blackmagic Desktop Video",
    ]
    
    found = False
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"✅ Blackmagic Desktop Video найден: {path}")
            found = True
            
            # Проверяем наличие утилит
            setup_exe = os.path.join(path, "Blackmagic Desktop Video Setup.exe")
            if os.path.exists(setup_exe):
                print(f"   - Desktop Video Setup: {setup_exe}")
            
            control_exe = os.path.join(path, "Blackmagic Desktop Video Control Panel.exe")
            if os.path.exists(control_exe):
                print(f"   - Control Panel: {control_exe}")
    
    if not found:
        print("⚠️ Blackmagic Desktop Video не найден в стандартных путях")
        print("   Установите Blackmagic Desktop Video для работы с DeckLink")
    
    return found

def print_routing_info():
    """Вывод информации о возможных причинах дублирования"""
    print("\n" + "="*70)
    print("🔍 ДИАГНОСТИКА МАРШРУТИЗАЦИИ DECKLINK")
    print("="*70)
    print("\n📋 ВОЗМОЖНЫЕ ПРИЧИНЫ ДУБЛИРОВАНИЯ КАМЕРЫ 0 НА ВЫХОДЫ 0 И 5:")
    print("\n1. ⚙️ Настройки Blackmagic Desktop Video:")
    print("   - Откройте 'Blackmagic Desktop Video Setup'")
    print("   - Проверьте настройки 'Video Output' или 'SDI Output'")
    print("   - Убедитесь, что не включен 'Mirroring' или 'Loopback'")
    print("   - Проверьте конфигурацию 'SDI Output Link Configuration'")
    print("\n2. 🎛️ Настройки пульта управления:")
    print("   - Если пульт идёт на 2 канала, проверьте маршрутизацию")
    print("   - Убедитесь, что вход 0 не назначен на выходы 0 и 5 одновременно")
    print("   - Проверьте настройки 'Input to Output Mapping'")
    print("\n3. 🔌 Аппаратные настройки DeckLink:")
    print("   - Некоторые DeckLink устройства имеют несколько выходов")
    print("   - Выходы могут быть настроены на один и тот же вход")
    print("   - Проверьте физические переключатели на устройстве (если есть)")
    print("\n4. 💻 Конфигурация системы:")
    print("   - Проверьте настройки Windows для устройств захвата")
    print("   - Убедитесь, что нет дублирования устройств в диспетчере устройств")
    print("="*70)

def main():
    print_routing_info()
    
    print("\n🔧 ПРОВЕРКА СИСТЕМЫ:")
    print("-" * 70)
    
    # Проверка Blackmagic Desktop Video
    print("\n1. Проверка Blackmagic Desktop Video...")
    check_blackmagic_desktop_video()
    
    # Проверка FFmpeg
    print("\n2. Проверка FFmpeg с поддержкой DeckLink...")
    check_ffmpeg_decklink()
    
    print("\n" + "="*70)
    print("💡 РЕКОМЕНДАЦИИ:")
    print("="*70)
    print("\n1. Откройте 'Blackmagic Desktop Video Setup' и проверьте:")
    print("   - Настройки выходов (Output Settings)")
    print("   - Конфигурацию SDI выходов (SDI Output Configuration)")
    print("   - Наличие функций зеркалирования/дублирования")
    print("\n2. Если используете пульт управления:")
    print("   - Проверьте настройки маршрутизации входов на выходы")
    print("   - Убедитесь, что вход 0 не дублируется на выходы 0 и 5")
    print("\n3. Проверьте документацию вашего DeckLink устройства:")
    print("   - Некоторые устройства имеют несколько выходов")
    print("   - Выходы могут быть настроены независимо или зеркалироваться")
    print("="*70)

if __name__ == "__main__":
    main()


