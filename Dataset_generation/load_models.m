function model_data = load_models(background)
    model_data.model_files = ["Cube","Cylinder","Sphere"]; %file paths to model .stl files
    for i = 1:length(model_data.model_files)
        [~, ~]=system(sprintf('mkdir %s_%s_dataset',model_data.model_files(i),background),'-echo');
        [f,~,~] = stlread(sprintf('Models/%s.stl',model_data.model_files(i)));
        model_data.model_objects.(model_data.model_files(i)) = f;
    end
end