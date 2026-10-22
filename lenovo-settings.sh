cd /home/swdhahn/apps/LenovoLegionLinux/kernel_module/

make clean
make
sudo make install
sudo depmod -a

sudo modprobe legion-laptop

sudo python3 /home/swdhahn/apps/LenovoLegionLinux/python/legion_linux/legion_linux/legion_gui.py

