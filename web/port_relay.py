"""
Relay TCP dari node interaktif (tempat VSCode Remote-SSH nyambung, mis.
mahoni01) ke port yang jalan di node compute SLURM (mis. "a100", tempat
web/run_web.sh jalan) -- TANPA butuh akses SSH langsung ke node compute
(publickey/password tidak di-auth di sana), cukup pakai `srun --overlap`
yang memang sudah diizinkan untuk job milik user sendiri.

Untuk tiap koneksi masuk ke <local_port> di node ini, script membuka
subprocess `srun --jobid=<id> --overlap socat - TCP:localhost:<remote_port>`
dan menyambungkan byte stdin/stdout subprocess itu ke socket-nya.

Supaya VSCode bisa forward port ini ke browser lokal, tinggal buka panel
"Ports" di VSCode -> Forward a Port -> isi <local_port>.

Cara pakai:
    python3 web/port_relay.py --jobid 26899 --port 5173
"""
import argparse
import selectors
import socket
import subprocess
import sys
import threading


def handle_conn(conn, jobid, remote_port):
    proc = subprocess.Popen(
        ["srun", f"--jobid={jobid}", "--overlap", "socat", "-", f"TCP:localhost:{remote_port}"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )

    def pump(src_read, dst_write, flush=None):
        try:
            while True:
                data = src_read(65536)
                if not data:
                    break
                dst_write(data)
                if flush:
                    flush()
        except Exception:
            pass

    # NOTE: proc.stdout.read(n) blocks until n bytes are buffered or EOF --
    # wrong for a live relay where the other side may only trickle a few
    # bytes at a time. read1(n) makes at most one underlying read() call and
    # returns whatever is available immediately. proc.stdin is a
    # BufferedWriter, so writes need an explicit flush to actually reach the
    # subprocess instead of sitting in Python's buffer.
    t1 = threading.Thread(
        target=pump, args=(conn.recv, proc.stdin.write, proc.stdin.flush), daemon=True
    )
    t2 = threading.Thread(target=pump, args=(proc.stdout.read1, conn.sendall), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    try:
        proc.stdin.close()
    except Exception:
        pass
    t2.join()
    proc.wait()
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobid", required=True)
    ap.add_argument("--port", type=int, required=True, help="port lokal & remote (sama)")
    ap.add_argument("--remote-port", type=int, default=None, help="kalau beda dari --port")
    args = ap.parse_args()
    remote_port = args.remote_port or args.port

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(20)
    print(f"Relay siap: 127.0.0.1:{args.port} -> job {args.jobid} -> localhost:{remote_port}", flush=True)

    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle_conn, args=(conn, args.jobid, remote_port), daemon=True).start()


if __name__ == "__main__":
    main()
