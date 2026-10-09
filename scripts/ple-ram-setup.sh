#!/bin/bash
# Copies the enrolled GDS PLE artifact onto a brd RAM block device for Q38_PLE_HOST_READ=1. Run as root; idempotent.
#   PLE_SOURCE    enrolled artifact directory (required)
#   PLE_IDENTITY  identity JSON written by `prepare-ple.py enroll` (required)
#   PLE_RAM_MOUNT mount point, default /mnt/ple-ram; the copy lands in $PLE_RAM_MOUNT/$(basename $PLE_SOURCE)
#   PLE_RAM_GIB   RAM disk size, default 52
set -euo pipefail
SRC=${PLE_SOURCE:?set PLE_SOURCE to the enrolled artifact directory}
IDENTITY=${PLE_IDENTITY:?set PLE_IDENTITY to the enrolled identity JSON}
MNT=${PLE_RAM_MOUNT:-/mnt/ple-ram}
DST=$MNT/$(basename "$SRC")
DEV=/dev/ram0
SIZE_KIB=$(( ${PLE_RAM_GIB:-52} * 1024 * 1024 ))
RULE=/etc/udev/rules.d/61-ple-ram.rules

identity_ok() {
  python3 - "$1" "$IDENTITY" <<"PY"
import hashlib, json, sys
from pathlib import Path
art, ident = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
gen = art / "generations" / ident["generation"]
ok = (sha(art / "CURRENT") == ident["current_sha256"]
      and sha(gen / "ple-gds-metadata.json") == ident["metadata_sha256"])
sys.exit(0 if ok else 1)
PY
}

drop_cache() {
  python3 - "$@" <<"PY"
import os, sys
for root in sys.argv[1:]:
    for d, _, files in os.walk(root):
        for f in files:
            fd = os.open(os.path.join(d, f), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
PY
}

# cuFile resolves the block device behind the file through udev, and 60-persistent-storage.rules skips ram*.
if [ ! -f "$RULE" ]; then
  echo 'SUBSYSTEM=="block", KERNEL=="ram0", ACTION!="remove", IMPORT{builtin}="blkid"' > "$RULE"
  udevadm control --reload
fi
udev_probe() { udevadm trigger --action=change --sysname-match=ram0 && udevadm settle; }

mkdir -p "$MNT"
if [ "$(findmnt -no SOURCE "$MNT" 2>/dev/null)" = "$DEV" ] && [ -d "$DST" ] && identity_ok "$DST" \
   && [ "$(du -sb --apparent-size "$DST" | cut -f1)" = "$(du -sb --apparent-size "$SRC" | cut -f1)" ]; then
  udev_probe
  echo "artifact already complete in $DST"; exit 0
fi

mountpoint -q "$MNT" && umount "$MNT"
if [ ! -b "$DEV" ]; then
  modprobe brd rd_nr=1 rd_size="$SIZE_KIB" max_part=0
  udevadm settle
fi
[ "$(($(cat /sys/block/ram0/size) / 2))" -ge "$SIZE_KIB" ] || { echo "$DEV is too small (brd already loaded with another size)"; exit 1; }

mkfs.ext4 -q -F -m 0 -O ^has_journal -E lazy_itable_init=0,lazy_journal_init=0 "$DEV"
mount -o noatime "$DEV" "$MNT"
echo "$(date -Is) copying $SRC -> $DST"
cp -a "$SRC" "$DST"
sync
identity_ok "$DST" || { echo "copy does not match $IDENTITY"; exit 1; }
drop_cache "$DST" "$SRC"
mount -o remount,ro "$MNT"
udev_probe
grep -q "ID_FS_USAGE=filesystem" /run/udev/data/b1:0 || { echo "udev does not report a filesystem on $DEV"; exit 1; }
echo "$(date -Is) done: $(du -sh "$DST" | cut -f1) on $DEV"
