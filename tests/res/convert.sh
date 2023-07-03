for d in *.dump ; do
    test_f=${d:0:(-5)}
    echo "${test_f}"
    riscv64-unknown-elf-objcopy -O binary ${test_f} "${test_f}.bin"
done