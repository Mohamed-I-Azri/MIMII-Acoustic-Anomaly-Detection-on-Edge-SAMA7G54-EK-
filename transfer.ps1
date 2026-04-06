foreach ($mid in @("id_00","id_02","id_04","id_06")) {
    foreach ($line in Get-Content "C:\Users\aazry\mimii_project\test_files_$mid.txt") {
        $line = $line.Trim()
        if ($line -match "abnormal") { $dst = "/data/mimii_test/$mid/abnormal/" }
        else { $dst = "/data/mimii_test/$mid/normal/" }
        scp -O -i "C:\Users\aazry\.ssh\sama7g54" -o IdentitiesOnly=yes -o ServerAliveInterval=10 $line "root@192.168.100.46:$dst"
    }
}
