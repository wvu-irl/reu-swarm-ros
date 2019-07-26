version=1.1.0
#########################################################
#							#
#	Compiles project				#
#	and Flashes all online particles		#
# 							#
#	Author: Henry Vos				#
#							#
#########################################################                         

rm -r ./build # deleting build folder
mkdir ./build # re-creating build folder
particle compile argon --target 1.1.0 # compiling project
mv argon_firmware_*.bin build # putting firmware into build

particle list online | while read DEVICE; do # Going through all online devices

	dev=$(echo $DEVICE | cut -d' ' -f1) # getting device name
	case "$dev" in # checking that device name is valid
		swarm_*)
			echo "Flashing $dev" # Flasing devices in separate threads
			particle flash --target $version $dev ./build/argon_firmware_*.bin &
	esac

done
wait # waiting for threads

