function par = parameter_builder(parfile)

    par = struct;
    fileID = fopen(parfile, 'r');
    tline = fgetl(fileID);
    while ischar(tline)
        tline = fgetl(fileID);
        if tline == -1
            break;
        end
    %     disp(tline);
        temp = split(tline);
        if size(temp, 1) == 1 || temp{1,1}(1) == '#'
            continue
        end
        if ~isnan(str2double(temp{2,1}))
            par.(temp{1,1}) = str2double(temp{2,1});
        else
            par.(temp{1,1}) = temp{2,1};
        end
    end
%     fclose(fileID);
%     save( parfile, "par");
end