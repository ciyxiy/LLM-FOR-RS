# debug_movies.py
"""调试movies.dat解析问题"""

import sys
import os
sys.path.append('.')

def test_file_parsing():
    """测试文件解析过程"""
    file_path = 'data/movies.dat'
    
    print("=== 基础文件检查 ===")
    print(f"文件存在: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        print("❌ 文件不存在!")
        return
    
    print(f"文件大小: {os.path.getsize(file_path)} bytes")
    
    print("\n=== 文件内容检查 ===")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"总行数: {len(lines)}")
        
        print("\n前5行内容:")
        for i, line in enumerate(lines[:5]):
            print(f"行 {i+1}: {repr(line)}")
            stripped = line.strip()
            if stripped:
                if '::' in stripped:
                    parts = stripped.split('::')
                    print(f"  分割后: {len(parts)} 部分")
                    for j, part in enumerate(parts):
                        print(f"    部分{j+1}: '{part}'")
                    try:
                        item_id = int(parts[0].strip())
                        print(f"  ID转换成功: {item_id}")
                    except Exception as e:
                        print(f"  ID转换失败: {e}")
                else:
                    print("  没有找到 '::' 分隔符")
            else:
                print("  空行")

def test_load_item_names():
    """测试物品名称加载"""
    print("\n=== 手动解析测试 ===")
    try:
        file_path = 'data/movies.dat'
        item_names = []
        item_ids = []
        
        print(f"开始加载: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > 5:  # 只处理前5行进行调试
                    break
                    
                line = line.strip()
                print(f"\n处理行 {line_num}: '{line}'")
                
                if not line:
                    print("  跳过空行")
                    continue
                
                try:
                    # movie.dat格式: ID::Title::Genres
                    parts = line.split('::')
                    print(f"  分割结果: {len(parts)} 部分: {parts}")
                    
                    if len(parts) >= 2:
                        item_id = int(parts[0].strip())
                        item_name = parts[1].strip()
                        
                        print(f"  解析结果 - ID: {item_id}, 名称: '{item_name}'")
                        
                        # 简单的类型添加
                        if len(parts) >= 3:
                            genres = parts[2].strip()
                            item_name = f"{item_name} [{genres}]"
                            print(f"  添加类型后: '{item_name}'")
                        
                        item_names.append(item_name)
                        item_ids.append(item_id)
                        print(f"  ✅ 成功添加物品")
                    else:
                        print(f"  ❌ 格式无效: 部分数量不足")
                        
                except ValueError as e:
                    print(f"  ❌ ID转换错误: {e}")
                except Exception as e:
                    print(f"  ❌ 其他错误: {e}")
        
        print(f"\n最终结果:")
        print(f"  加载了 {len(item_names)} 个物品")
        print(f"  前3个物品: {item_names[:3]}")
        
    except Exception as e:
        print(f"加载测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("�� 开始调试movies.dat解析问题...")
    
    test_file_parsing()
    test_load_item_names()
    
    print("\n�� 调试完成!")
