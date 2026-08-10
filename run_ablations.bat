@echo off
setlocal

echo Backing up your current (all-4-changes) src files...
copy /Y src\elo.py ablation_variants\elo_FULL_backup.py >nul
copy /Y src\ml_model.py ablation_variants\ml_model_FULL_backup.py >nul

echo.
echo === Run A: elo regression-to-mean ONLY ===
copy /Y ablation_variants\elo_A_regression_only.py src\elo.py >nul
copy /Y ablation_variants\ml_model_ORIGINAL.py src\ml_model.py >nul
python -m src.backtest > ablation_A_regression_only.txt
echo   -> saved ablation_A_regression_only.txt

echo.
echo === Run B: home/away Elo split ONLY ===
copy /Y ablation_variants\elo_B_split_only.py src\elo.py >nul
copy /Y ablation_variants\ml_model_ORIGINAL.py src\ml_model.py >nul
python -m src.backtest > ablation_B_split_only.txt
echo   -> saved ablation_B_split_only.txt

echo.
echo === Run C: recency-weighted training rows ONLY ===
copy /Y ablation_variants\elo_ORIGINAL.py src\elo.py >nul
copy /Y ablation_variants\ml_model_C_recency_only.py src\ml_model.py >nul
python -m src.backtest > ablation_C_recency_only.txt
echo   -> saved ablation_C_recency_only.txt

echo.
echo === Run D: early stopping ONLY ===
copy /Y ablation_variants\elo_ORIGINAL.py src\elo.py >nul
copy /Y ablation_variants\ml_model_D_earlystop_only.py src\ml_model.py >nul
python -m src.backtest > ablation_D_earlystop_only.txt
echo   -> saved ablation_D_earlystop_only.txt

echo.
echo Restoring your updated (all-4-changes) files...
copy /Y ablation_variants\elo_FULL_backup.py src\elo.py >nul
copy /Y ablation_variants\ml_model_FULL_backup.py src\ml_model.py >nul

echo.
echo All done. Four files written to this folder:
echo   ablation_A_regression_only.txt
echo   ablation_B_split_only.txt
echo   ablation_C_recency_only.txt
echo   ablation_D_earlystop_only.txt
echo Your src\ folder is back to the full (all 4 changes) version.
endlocal
