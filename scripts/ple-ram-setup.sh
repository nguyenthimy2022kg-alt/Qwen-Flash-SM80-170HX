#!/bin/bash
# Create a dedicated brd device, or verify a previously completed setup.
# PLE_RAM_RESERVE_GIB leaves RAM for the model and OS (default 24 GiB).
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run as root.' >&2; exit 1; }
SRC=$(realpath -e "${PLE_SOURCE:?set PLE_SOURCE}")
IDENTITY=$(realpath -e "${PLE_IDENTITY:?set PLE_IDENTITY}")
MNT=$(realpath -m "${PLE_RAM_MOUNT:-/mnt/ple-ram}")
DST=$MNT/$(basename "$SRC")
DEV=/dev/ram0
RULE=/etc/udev/rules.d/61-ple-ram.rules
OWNER=/run/qwen-ple-ram.owner
HELPER=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/ple-ram-artifact.py
GIB=${PLE_RAM_GIB:-52}
RESERVE=${PLE_RAM_RESERVE_GIB:-24}
[[ $GIB =~ ^[1-9][0-9]{0,3}$ && $RESERVE =~ ^[0-9]{1,4}$ ]] || { echo 'Invalid GiB setting.' >&2; exit 1; }
RESERVE=$((10#$RESERVE))
SIZE_KIB=$(( GIB * 1024 * 1024 ))
exec 9>/run/qwen-ple-ram.lock
flock -n 9 || { echo 'Another RAM-disk setup is running.' >&2; exit 1; }
[[ -f $HELPER && -d $SRC && $SRC != "$MNT" && $SRC != "$MNT/"* && $MNT != "$SRC/"* ]] || { echo 'Invalid source, destination or missing helper.' >&2; exit 1; }
SOURCE_BYTES=$(python3 "$HELPER" size "$SRC" "$IDENTITY")
BOOT=$(cat /proc/sys/kernel/random/boot_id)

# The record is written only after a verified read-only mount. An existing
# unowned ram0 or brd module is never reformatted or unloaded.
if [[ -e /sys/module/brd || -e $DEV ]]; then
  [[ -f $OWNER && ! -L $OWNER ]] || { echo 'Existing brd/ram0 is not owned by this setup; refusing to touch it.' >&2; exit 1; }
  UUID=$(blkid -s UUID -o value "$DEV")
  EXPECTED=$(printf '%s\n%s\n%s\n%s\n%s\n' "$BOOT" "$DEV" "$MNT" "$UUID" "$SRC")
  [[ $(cat "$OWNER") == "$EXPECTED" ]] || { echo 'RAM-disk ownership does not match; refusing to touch it.' >&2; exit 1; }
  [[ $(findmnt -rn -M "$MNT" -o SOURCE) == "$DEV" ]] || { echo 'Owned RAM disk is not mounted as expected.' >&2; exit 1; }
  OPTIONS=$(findmnt -rn -M "$MNT" -o OPTIONS)
  [[ ,$OPTIONS, == *,ro,* ]] || { echo 'Owned RAM disk must be read-only.' >&2; exit 1; }
  python3 "$HELPER" verify "$DST" "$IDENTITY"
  echo "Verified existing artifact in $DST"; exit 0
fi
if mountpoint -q "$MNT" || [[ -d $MNT && -n $(ls -A "$MNT") ]]; then
  echo 'Mount point is already mounted or nonempty; refusing to cover existing data.' >&2; exit 1
fi
AVAILABLE_KIB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
NEEDED_KIB=$(( (SOURCE_BYTES + 1023) / 1024 + (RESERVE + 1) * 1024 * 1024 ))
[[ $AVAILABLE_KIB -ge $NEEDED_KIB ]] || { echo "Not enough free RAM for the artifact plus ${RESERVE} GiB reserve and filesystem overhead." >&2; exit 1; }
[[ $((SIZE_KIB * 1024)) -gt $((SOURCE_BYTES + 1024 * 1024 * 1024)) ]] || { echo 'RAM disk is too small for payload plus filesystem overhead.' >&2; exit 1; }
EXPECTED_RULE='SUBSYSTEM=="block", KERNEL=="ram0", ACTION!="remove", IMPORT{builtin}="blkid"'
if [[ -e $RULE && $(cat "$RULE") != "$EXPECTED_RULE" ]]; then
  echo "Existing $RULE has different contents; refusing to overwrite it." >&2; exit 1
fi

mkdir -p "$MNT"
# Failed setup is left for explicit inspection, never reformatted on retry.
modprobe brd rd_nr=1 rd_size="$SIZE_KIB" max_part=0
udevadm settle
[[ -b $DEV && $(($(cat /sys/block/ram0/size) / 2)) -ge $SIZE_KIB ]] || { echo 'Unexpected RAM-device size.' >&2; exit 1; }
[[ -z $(findmnt -rn -S "$DEV") ]] || { echo 'RAM device is already mounted; refusing to format.' >&2; exit 1; }
mkfs.ext4 -q -F -m 0 -O ^has_journal -E lazy_itable_init=0,lazy_journal_init=0 "$DEV"
mount -o noatime "$DEV" "$MNT"
python3 "$HELPER" copy "$SRC" "$IDENTITY" "$DST"
sync -f "$MNT"
mount -o remount,ro "$MNT"
printf '%s\n' "$EXPECTED_RULE" > "$RULE"
udevadm control --reload
udevadm trigger --action=change --sysname-match=ram0
udevadm settle
grep -q 'ID_FS_USAGE=filesystem' /run/udev/data/b1:0 || { echo 'udev does not report the RAM filesystem.' >&2; exit 1; }
UUID=$(blkid -s UUID -o value "$DEV")
printf '%s\n%s\n%s\n%s\n%s\n' "$BOOT" "$DEV" "$MNT" "$UUID" "$SRC" > "$OWNER"
echo "Verified artifact ready in $DST ($DEV, read-only)"
