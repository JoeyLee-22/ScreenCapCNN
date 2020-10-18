while true; do
    printf "\n"
    read -p "Clear All? (y/n) " yn
    if [[ "$yn" == "y" ]]; then
        rm dataset/*
        rm test_images/*
        rm train_images/*
        break
    elif [[ "$yn" == "n" ]]; then
        while true; do
            printf "\n"
            read -p "Clear train_images? (y/n) " train_images_choice
            if [[ "$train_images_choice" == "y" ]]; then
                rm train_images/*
                break
            elif [[ "$train_images_choice" == "n" ]]; then
                break
            fi
        done        
        while true; do
            printf "\n"
            read -p "Clear test_images? (y/n) " test_images_choice
            if [[ "$test_images_choice" == "y" ]]; then
                rm test_images/*
                break
            elif [[ "$test_images_choice" == "n" ]]; then
                break
            fi
        done
        rm dataset/*
        break
    fi
done