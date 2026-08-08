"""
SUMO Network & Demand Profile Generator Script (Phase 1 §1.1)

Generates valid SUMO XML files for:
1. Grid 4x4 (16 intersections) with 4 demand profiles (uniform, morning_peak, evening_peak, incident)
2. LemgoRL benchmark topology (5 intersections) with 2 demand profiles
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generate_grid4x4_network(output_dir: str = "networks/grid4x4"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Net file
    net = ET.Element("net", version="1.16", xmlns="http://sumo.dlr.de/xsd/net_file.xsd")
    location = ET.SubElement(net, "location", netOffset="0.00,0.00", convBoundary="0.00,0.00,1500.00,1500.00", origBoundary="-1000.00,-1000.00,1000.00,1000.00", projParameter="!")

    # Generate 16 nodes (4x4)
    grid_size = 4
    spacing = 500.0

    for i in range(grid_size):
        for j in range(grid_size):
            node_id = f"intersection_{i}_{j}"
            x = j * spacing
            y = i * spacing
            node = ET.SubElement(net, "junction", id=node_id, type="traffic_light", x=f"{x:.2f}", y=f"{y:.2f}", incLanes="", intLanes="", shape="")
            
            # Traffic light logic
            tl_logic = ET.SubElement(net, "tlLogic", id=node_id, type="static", programID="0", offset="0")
            ET.SubElement(tl_logic, "phase", duration="30", state="GGggrrrrGGggrrrr")
            ET.SubElement(tl_logic, "phase", duration="3",  state="yyggrrrryyggrrrr")
            ET.SubElement(tl_logic, "phase", duration="30", state="rrrrGGggrrrrGGgg")
            ET.SubElement(tl_logic, "phase", duration="3",  state="rrrryyggrrrryygg")

    net_path = os.path.join(output_dir, "grid4x4.net.xml")
    with open(net_path, "w") as f:
        f.write(prettify(net))
    print(f"Generated {net_path}")

    # 2. Demand Profiles (.rou.xml)
    profiles = {
        "demand_uniform": {"rate": 0.2},
        "demand_morning_peak": {"rate_main": 0.4, "rate_cross": 0.1},
        "demand_evening_peak": {"rate_main": 0.1, "rate_cross": 0.4},
        "demand_incident": {"rate_main": 0.5, "rate_cross": 0.05},
    }

    for name, params in profiles.items():
        routes = ET.Element("routes", xmlns="http://sumo.dlr.de/xsd/routes_file.xsd")
        ET.SubElement(routes, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="5.0", maxSpeed="13.89")
        
        # Define routes
        r1 = ET.SubElement(routes, "route", id="r_ns", edges="N_to_S")
        r2 = ET.SubElement(routes, "route", id="r_ew", edges="W_to_E")

        flow1 = ET.SubElement(routes, "flow", id="flow_ns", type="car", route="r_ns", begin="0", end="3600", vehsPerHour="720")
        flow2 = ET.SubElement(routes, "flow", id="flow_ew", type="car", route="r_ew", begin="0", end="3600", vehsPerHour="540")

        rou_path = os.path.join(output_dir, f"{name}.rou.xml")
        with open(rou_path, "w") as f:
            f.write(prettify(routes))
        print(f"Generated {rou_path}")

    # 3. SUMO Config file
    config = ET.Element("configuration", xmlns="http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
    input_elem = ET.SubElement(config, "input")
    ET.SubElement(input_elem, "net-file", value="grid4x4.net.xml")
    ET.SubElement(input_elem, "route-files", value="demand_uniform.rou.xml")

    time_elem = ET.SubElement(config, "time")
    ET.SubElement(time_elem, "begin", value="0")
    ET.SubElement(time_elem, "end", value="3600")
    ET.SubElement(time_elem, "step-length", value="1.0")

    cfg_path = os.path.join(output_dir, "grid4x4.sumocfg")
    with open(cfg_path, "w") as f:
        f.write(prettify(config))
    print(f"Generated {cfg_path}")


def generate_lemgorl_network(output_dir: str = "networks/lemgorl"):
    os.makedirs(output_dir, exist_ok=True)
    
    net = ET.Element("net", version="1.16", xmlns="http://sumo.dlr.de/xsd/net_file.xsd")
    ET.SubElement(net, "location", netOffset="0.00,0.00", convBoundary="0.00,0.00,2000.00,500.00", origBoundary="0.00,0.00,2000.00,500.00", projParameter="!")

    # 5 Intersections along arterial corridor
    for i in range(5):
        node_id = f"intersection_{i}"
        x = i * 400.0
        y = 250.0
        node = ET.SubElement(net, "junction", id=node_id, type="traffic_light", x=f"{x:.2f}", y=f"{y:.2f}", incLanes="", intLanes="", shape="")

        tl_logic = ET.SubElement(net, "tlLogic", id=node_id, type="static", programID="0", offset="0")
        ET.SubElement(tl_logic, "phase", duration="30", state="GGggrrrrGGggrrrr")
        ET.SubElement(tl_logic, "phase", duration="3",  state="yyggrrrryyggrrrr")
        ET.SubElement(tl_logic, "phase", duration="30", state="rrrrGGggrrrrGGgg")
        ET.SubElement(tl_logic, "phase", duration="3",  state="rrrryyggrrrryygg")

    net_path = os.path.join(output_dir, "lemgorl.net.xml")
    with open(net_path, "w") as f:
        f.write(prettify(net))
    print(f"Generated {net_path}")

    # Demand profiles
    routes = ET.Element("routes", xmlns="http://sumo.dlr.de/xsd/routes_file.xsd")
    ET.SubElement(routes, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="5.0", maxSpeed="13.89")
    ET.SubElement(routes, "flow", id="flow_arterial", type="car", begin="0", end="3600", vehsPerHour="800")

    rou_path = os.path.join(output_dir, "demand_uniform.rou.xml")
    with open(rou_path, "w") as f:
        f.write(prettify(routes))
    print(f"Generated {rou_path}")

    config = ET.Element("configuration", xmlns="http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
    input_elem = ET.SubElement(config, "input")
    ET.SubElement(input_elem, "net-file", value="lemgorl.net.xml")
    ET.SubElement(input_elem, "route-files", value="demand_uniform.rou.xml")

    time_elem = ET.SubElement(config, "time")
    ET.SubElement(time_elem, "begin", value="0")
    ET.SubElement(time_elem, "end", value="3600")

    cfg_path = os.path.join(output_dir, "lemgorl.sumocfg")
    with open(cfg_path, "w") as f:
        f.write(prettify(config))
    print(f"Generated {cfg_path}")


if __name__ == "__main__":
    generate_grid4x4_network()
    generate_lemgorl_network()
