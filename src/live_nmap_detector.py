from scapy.all import sniff, IP, TCP
from collections import defaultdict, deque
import time
import threading

# Stores recent SYN packets per source IP
syn_tracker = defaultdict(lambda: deque())

# Shared list of live alerts for Streamlit
live_alerts = []

# Detection settings (adjust if needed)
TIME_WINDOW = 5          # seconds
PORT_THRESHOLD = 5       # lower = easier to detect Nmap
PACKET_THRESHOLD = 8     # lower = easier to detect Nmap

# Cooldown to prevent spam alerts from same source
last_alert_time = {}
ALERT_COOLDOWN = 5       # seconds

def process_packet(packet):
    global live_alerts

    try:
        if packet.haslayer(IP) and packet.haslayer(TCP):
            ip_layer = packet[IP]
            tcp_layer = packet[TCP]

            # Detect SYN packets only (no ACK)
            if tcp_layer.flags == "S":
                src_ip = ip_layer.src
                dst_ip = ip_layer.dst
                dst_port = tcp_layer.dport
                now = time.time()

                # Add packet to tracker
                syn_tracker[src_ip].append((now, dst_port, dst_ip))

                # Remove old packets outside the time window
                while syn_tracker[src_ip] and now - syn_tracker[src_ip][0][0] > TIME_WINDOW:
                    syn_tracker[src_ip].popleft()

                recent_packets = list(syn_tracker[src_ip])
                unique_ports = len(set(p[1] for p in recent_packets))
                total_syns = len(recent_packets)

                # Check for scan behavior
                if unique_ports >= PORT_THRESHOLD or total_syns >= PACKET_THRESHOLD:
                    # Cooldown check
                    if src_ip in last_alert_time and (now - last_alert_time[src_ip] < ALERT_COOLDOWN):
                        return

                    last_alert_time[src_ip] = now

                    alert = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "unique_ports": unique_ports,
                        "syn_packets": total_syns,
                        "alert": "Possible Nmap Port Scan"
                    }

                    live_alerts.append(alert)

    except Exception:
        # Prevent sniff thread from crashing
        pass

def start_sniffer(interface=None):
    sniff(
        filter="tcp",
        prn=process_packet,
        store=False,
        iface=interface
    )

def run_sniffer_in_thread(interface=None):
    thread = threading.Thread(target=start_sniffer, args=(interface,), daemon=True)
    thread.start()